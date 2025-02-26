from MMLLM_Claude import MMLLM_Claude
from MMLLM_Gemini import MMLLM_Gemini
from MMLLM_GPT import MMLLM_GPT
from MMLLM_Common import InputDataset


if __name__ == "__main__":

    claude_exp = MMLLM_Claude('<cluade_api_key>')
    claude_exp.phase1_brand_identification(InputDataset.MMLLM_Benign)
    claude_exp.phase2_phishing_classification(InputDataset.MMLLM_Benign)

    gpt_exp = MMLLM_GPT('<gpt_api_key>')
    gpt_exp.phase1_brand_identification(InputDataset.MMLLM_Phishing)
    gpt_exp.phase2_phishing_classification(InputDataset.MMLLM_Phishing)

    gemini_exp = MMLLM_Gemini('<gpt_api_key>')
    gemini_exp.phase1_brand_identification(InputDataset.MMLLM_Benign)
    gemini_exp.phase2_phishing_classification(InputDataset.MMLLM_Benign)


    