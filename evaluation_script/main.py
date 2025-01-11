import json
import numpy as np
import scipy
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import tqdm
import time
import copy
import properties
import google.generativeai as genai

def pairwise_meteor(candidate, reference):
    return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))


def compute_all_pairwise_scores(src_data, tgt_data, metric):
    scores = np.empty((len(src_data), len(tgt_data)))

    for i, src in enumerate(src_data):
        for j, tgt in enumerate(tgt_data):
            scores[i][j] = metric(src, tgt)

    return scores


class EV2REvaluator:

    verdicts = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]
    # Config
    _API = properties.ModelApi.GEMINI_FLASH  # .GPT4o
    _PROMPT_TYPE = properties.PromptTypes("atomic_reference_prec_recall")
    _OUTPUT_FILE = "predictions_{}_{}.jsonl".format(_PROMPT_TYPE.value, _API.value)
    _CORRELATION_OUPUT_FILE = "correlation_{}_{}.csv".format(_PROMPT_TYPE.value, _API.value)
    prompt_type = properties.PromptTypes("atomic_reference_prec_recall")
    MAX_RETRIES = 10
    ev2r_reporting_levels = [0.46]  # [0.44, 0.46, 0.48]
    # LLM
    MAX_TOKENS = 3000
    TEMPERATURE = 0
    #
    GEMINI_KEY = "AIzaSyClJFx89pJR0_8yc1nvTClMUzFPj0r1dHA"
    # genai.configure(api_key=os.environ["API_KEY"])
    genai.configure(api_key=GEMINI_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-pro", generation_config={"response_mime_type": "application/json"})

    def prepare_dataset(self, srcs, tgts):
        srcs_data = []
        tgts_data = []

        for src, tgt in zip(srcs, tgts):
            #
            prediction_evidence = ""
            for src_qa in src['evidence']:
                prediction_evidence += "Question: " + src_qa["question"] + "\n" + "Answer: " + src_qa["answer"] + "\n\n"
            #
            reference_evidence = ""
            for tgt_qa in tgt['questions']:
                reference_evidence += "Question: " + tgt_qa["question"] + "\n" + "Answer: " + tgt_qa["answers"][0]["answer"] + "\n\n"

            if 'claim_id' not in tgt.keys():
                tgt['claim_id'] = src['claim_id']

            srcs_data.append(properties.AveritecEntry(claim=src['claim'],
                                                       label=src['pred_label'],
                                                       evidence=prediction_evidence,
                                                       id=src['claim_id']
                                                       ))
            tgts_data.append(properties.AveritecEntry(claim=tgt['claim'],
                                                        label=tgt['label'],
                                                        evidence=reference_evidence,
                                                        id=tgt['claim_id']
                                                        ))

        return srcs_data, tgts_data

    def query_gemini(self, prompt):
        try:
            return self.GEMINI_MODEL.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            candidate_count=1,
                            max_output_tokens=self.MAX_TOKENS,
                            temperature=self.TEMPERATURE,
                        ),
            )
        except Exception as e:
            print(e)
            return ""

    def prepare_prompt(self, tgt_sample, pred_sample):
        """Formats prompt using dataset sample as input."""
        prompt = properties.PROMPT_MAPPING[self.prompt_type].format(tgt_sample.claim,
                                                                    tgt_sample.evidence,
                                                                    pred_sample.evidence)
        return prompt

    def get_response_text(self, response):
        if type(response) == genai.types.generation_types.GenerateContentResponse:
            try:
                return response.text
            except Exception as e:
                print("Error in extracting Gemini response: {}".format(e))
                return ""

    def process_output(self, sample, response):
        logprob_inp = None
        return properties.OpenAIResponse(claim=sample.claim, evidence=sample.evidence,
                                         response=self.get_response_text(response),
                                         gold=sample.label.lower(), id=sample.id,
                                         logprobs=logprob_inp)

    def calculate_atomic_score_prec_recall_openai_response(self, response_llm):
        response_openai_copy = copy.deepcopy(response_llm)
        try:
            if type(response_llm.response) == str:
                response = json.loads(
                    response_llm.response.replace(": '", ": \"").replace("',", "\",").replace("':", "\":"))
            else:
                response = response_llm.response
            response_openai_copy.response = response
            response_openai_copy.response['precision'] = response["support predicted evidence"] / response[
                "facts count predicted evidence"]
            response_openai_copy.response['recall'] = response["support reference evidence"] / response[
                "facts count reference evidence"]
        except Exception as e:
            print("Following exception occurred: {}".format(e))
            return None

        return response_openai_copy

    def calculate_prediction_scores(self, responses):
        predictions_w_scores = []

        for i, res in enumerate(responses):
            pred_w_scores = self.calculate_atomic_score_prec_recall_openai_response(res)
            if pred_w_scores:
                predictions_w_scores.append(pred_w_scores)

        return predictions_w_scores

    def prompt_api_model(self, srcs, tgts):
        responses = []

        for i, tgt_sample in tqdm.tqdm(enumerate(tgts), desc="feed the prompt_atomic_reference_p_r to api model ..."):
            pred_sample = srcs[i]
            #
            prompt = self.prepare_prompt(tgt_sample, pred_sample)
            #
            attempt = 0
            while attempt < self.MAX_RETRIES:
                try:
                    response = self.query_gemini(prompt)
                    responses.append(self.process_output(tgt_sample, response))
                    print("One request successfully processed..")
                    break
                except:
                    attempt += 1
                    wait_time = 10 ** attempt  # Exponential backoff
                    print(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        return responses


    def evaluate_ev2r_score_ori(self, srcs, tgts, ev2r_scores):
        scores = []
        veracity_scores = []
        for src, tgt, ev2r_score in tqdm.tqdm(zip(srcs, tgts, ev2r_scores)):
            precision, recall = ev2r_score.response['precision'], ev2r_score.response['recall']

            this_example_scores = [0.0 for _ in self.ev2r_reporting_levels]
            for i, level in enumerate(self.ev2r_reporting_levels):
                if recall > level:
                    this_example_scores[i] = src["pred_label"] == tgt["label"]

            scores.append(this_example_scores)
            #
            veracity_scores.append(1.0 if src["pred_label"] == tgt["label"] else 0.0)

        return np.mean(np.array(scores), axis=0)

    def evaluate_ev2r_score(self, srcs, tgts, ev2r_scores):
        scores = []
        ev2r_evi_recall = []
        veracity_scores = []

        for i, (src, tgt) in enumerate(tqdm.tqdm(zip(srcs, tgts))):
            #
            this_example_scores = [0.0 for _ in self.ev2r_reporting_levels]
            for ev2r_score in ev2r_scores:
                if ev2r_score.id == i:
                    precision, recall = ev2r_score.response['precision'], ev2r_score.response['recall']
                    #
                    for j, level in enumerate(self.ev2r_reporting_levels):
                        if recall > level:
                            this_example_scores[j] = src["pred_label"] == tgt["label"]

                    scores.append(this_example_scores)
                    ev2r_evi_recall.append(recall)
                    veracity_scores.append(1.0 if src["pred_label"] == tgt["label"] else 0.0)
                    break
                if ev2r_score.id > i:
                    break

            if len(scores) != (i+1):
                scores.append(this_example_scores)
                ev2r_evi_recall.append(0.0)

        return np.mean(np.array(scores), axis=0)


class AVeriTeCEvaluator:

    verdicts = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]
    pairwise_metric = None
    max_questions = 10
    metric = None
    # averitec_reporting_levels = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    averitec_reporting_levels = [0.25]

    def __init__(self, metric="meteor"):
        self.metric = metric
        if metric == "meteor":
            self.pairwise_metric = pairwise_meteor

    def evaluate_averitec_score(self, srcs, tgts):
        scores = []
        for src, tgt in tqdm.tqdm(zip(srcs, tgts)):
            score = self.compute_pairwise_evidence_score(src, tgt)

            this_example_scores = [0.0 for _ in self.averitec_reporting_levels]
            for i, level in enumerate(self.averitec_reporting_levels):
                if score > level:
                    this_example_scores[i] = src["pred_label"] == tgt["label"]

            scores.append(this_example_scores)

        return np.mean(np.array(scores), axis=0)


    def evaluate_questions_only(self, srcs, tgts):
        all_utils = []
        for src, tgt in tqdm.tqdm(zip(srcs, tgts)):
            if "evidence" not in src:
                # If there was no evidence, use the string evidence
                src_questions = self.extract_full_comparison_strings(
                    src, is_target=False
                )[: self.max_questions]
            else:
                src_questions = [
                    qa["question"] for qa in src["evidence"][: self.max_questions]
                ]
            tgt_questions = [qa["question"] for qa in tgt["questions"]]

            pairwise_scores = compute_all_pairwise_scores(
                src_questions, tgt_questions, self.pairwise_metric
            )

            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )

            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_questions))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils)


    def compute_pairwise_evidence_score(self, src, tgt):
        """Different key is used for reference_data and prediction.
        For the prediction, the format is
        {"evidence": [
            {
                "question": "What does the increased federal medical assistance percentage mean for you?",
                "answer": "Appendix A: Applicability of the Increased Federal Medical Assistance Percentage ",
                "url": "https://www.medicaid.gov/federal-policy-guidance/downloads/smd21003.pdf"
            }],
        "pred_label": "Supported"}
        And for the data with fold label:
        {"questions": [
            {
                "question": "Where was the claim first published",
                "answers": [
                    {
                        "answer": "It was first published on Sccopertino",
                        "answer_type": "Abstractive",
                        "source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/",
                        "source_medium": "Web text",
                        "cached_source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"
                    }
                ]
            }]
        "label": "Refuted"}
        """

        src_strings = self.extract_full_comparison_strings(src, is_target=False)[
            : self.max_questions
        ]
        tgt_strings = self.extract_full_comparison_strings(tgt)
        pairwise_scores = compute_all_pairwise_scores(
            src_strings, tgt_strings, self.pairwise_metric
        )
        assignment = scipy.optimize.linear_sum_assignment(
            pairwise_scores, maximize=True
        )
        assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

        # Reweight to account for unmatched target questions
        reweight_term = 1 / float(len(tgt_strings))
        assignment_utility *= reweight_term
        return assignment_utility


    def evaluate_questions_and_answers(self, srcs, tgts):
        all_utils = []
        for src, tgt in tqdm.tqdm(zip(srcs, tgts)):
            src_strings = self.extract_full_comparison_strings(src, is_target=False)[
                : self.max_questions
            ]
            tgt_strings = self.extract_full_comparison_strings(tgt)

            pairwise_scores = compute_all_pairwise_scores(
                src_strings, tgt_strings, self.pairwise_metric
            )

            assignment = scipy.optimize.linear_sum_assignment(
                pairwise_scores, maximize=True
            )

            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_strings))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils)

    def extract_full_comparison_strings(self, example, is_target=True):
        example_strings = []

        if is_target:
            if "questions" in example:
                for evidence in example["questions"]:
                    # If the answers is not a list, make them a list:
                    if not isinstance(evidence["answers"], list):
                        evidence["answers"] = [evidence["answers"]]

                    for answer in evidence["answers"]:
                        example_strings.append(
                            evidence["question"] + " " + answer["answer"]
                        )
                        if (
                            "answer_type" in answer
                            and answer["answer_type"] == "Boolean" and "boolean_explanation" in answer
                        ):
                            example_strings[-1] += ". " + answer["boolean_explanation"]
                    if len(evidence["answers"]) == 0:
                        example_strings.append(
                            evidence["question"] + " No answer could be found."
                        )
        else:
            if "evidence" in example:
                for evidence in example["evidence"]:
                    example_strings.append(
                        evidence["question"] + " " + evidence["answer"]
                    )

        if "string_evidence" in example:
            for full_string_evidence in example["string_evidence"]:
                example_strings.append(full_string_evidence)
        return example_strings


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            "status": u"running",
            "when_made_public": None,
            "participant_team": 5,
            "input_file": "https://abc.xyz/path/to/submission/file.json",
            "execution_time": u"123",
            "publication_url": u"ABC",
            "challenge_phase": 1,
            "created_by": u"ABC",
            "stdout_file": "https://abc.xyz/path/to/stdout/file.json",
            "method_name": u"Test",
            "stderr_file": "https://abc.xyz/path/to/stderr/file.json",
            "participant_team_name": u"Test Team",
            "project_url": u"http://foo.bar",
            "method_description": u"ABC",
            "is_public": False,
            "submission_result_file": "https://abc.xyz/path/result/file.json",
            "id": 123,
            "submitted_at": u"2017-03-20T19:22:03.880652Z",
        }
    """
    print(kwargs["submission_metadata"])

    with open(user_submission_file) as f:
        predictions = json.load(f)[:2]

    with open(test_annotation_file) as f:
        references = json.load(f)[:2]

    # AVeriTeC scorer
    # scorer = AVeriTeCEvaluator()
    # Q_evidence_score = scorer.evaluate_questions_only(predictions, references)
    # QA_evidence_score = scorer.evaluate_questions_and_answers(predictions, references)
    # averitec_scores = scorer.evaluate_averitec_score(predictions, references)

    # EV2R scorer
    EV2R_scorer = EV2REvaluator()
    #
    start_time = time.time()
    pred_data, ref_data = EV2R_scorer.prepare_dataset(predictions, references)
    responses = EV2R_scorer.prompt_api_model(pred_data, ref_data)
    ev2r_scores = EV2R_scorer.calculate_prediction_scores(responses)

    averitec_ev2r_scores = EV2R_scorer.evaluate_ev2r_score(predictions, references, ev2r_scores)
    print("EV2R time: {}".format(time.time() - start_time))

    output = {}

    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "dev_split": {
                    "EV2R Score": averitec_ev2r_scores[0],  # (recall @ 0.46)
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["dev_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test" or phase_codename == "after_test" or phase_codename == "after_test_new_KB":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "EV2R Score": averitec_ev2r_scores[0],  # (recall @ 0.46)
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")

    return output


# if __name__ == "__main__":
#     # filename = 'test'   # test, dev
#     # test_annotation_file = "annotations/averitec_{}_gold.json".format(filename)
#     # user_submission_file = "HerO_{}_pred.json".format(filename)       # "yulong_{}_pred.json"
#
#     filename = 'dev'   # test, dev
#     test_annotation_file = "annotations/averitec_{}_gold.json".format(filename)
#     user_submission_file = "rami_50s_{}_pred.json".format(filename)       # "yulong_{}_pred.json"
#
#     evaluate(test_annotation_file, user_submission_file, filename)
#     print("hello")

