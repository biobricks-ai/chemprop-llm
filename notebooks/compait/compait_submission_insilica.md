CoMPAIT Model Submission Documentation
Initial Model Description

# General information
Date of submission:	01 Oct 2024
Group name: Insilica
Group leader: Dr. Thomas Luechtefeld
Model Developer(s): OpenAI & Insilica
Model Information:
	Endpoint modeled: mg/L
	Modeling algorithm: prompt engineering on gpt4
	Descriptors Software: none
	Selection Algorithm: none
	Selected descriptors: none

# Internal Validation
No training was done, we simply asked gpt4 to predict acute inhalation lc50 values. We thought this might make a good benchmark for other submissions. 

The openai structured-outputs tool, platform.openai.com/docs/guides/structured-outputs, allowed us to ask gpt4 to output a JSON structure. We asked it to output a numeric value for lc50 in mg/L and a confidence score between 1 and 10 with 10 being the highest confidence.

# Applicability Domain
We set the applicability domain to 1 (include) when confidence was 7 or higher.

Method used to assess the applicability domain: we evaluated performance at different confidence thresholds on the train set. 

Software for applicability domain assessment: none
Limits of applicability: difficult to define, but self-reported by the gpt-4 model. 
Mechanistic Interpretation (if possible): none
Mechanistic basis of the model: none
Additional information regarding the mechanistic interpretation: none
Supporting Information: none
Comments: please ignore `predictions_insilica.csv`, it was an accidental upload.
Supporting figures: none
Supporting tables: none

# FILES
- compait_submission_insilica.md: this file
- predictions_insilica.csv: accidental upload
- predictions_insilica_v2.csv: correct submission
- training_insilica.csv: training set evaluation
