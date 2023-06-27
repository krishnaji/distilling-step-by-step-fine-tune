
## Distilling Step-By-Step: Classify input as Enquiry or Non-Enquiry 

This is a technical reference implementation of a Distilling Step-by-Step(DSS) mechanism proposed in this paper(arXiv:2305.02301). The  DSS mechanism leverages the ability of LLMs to reason about their predictions to train smaller models in a data-efficient way.

The Vertex AI PaLM text-bison model is used to generate classification and rationale for fine tuning  the smaller model (flan-t5-base). 

The reference implementation uses Huggingface Transformers and Deepspeed libraries to fine tune the smaller model efficiently. 

### Train your own

Create a virtual  Environment and activate
``` 
python3 -m venv .venv
source .vnev/bin/activate
```


Install required libraries
```
pip install -r requirements.txt
```

Generate classification and rationale dataset using the LLM. We use Vertex AI PaLM text-bison LLM. Note that when we create the data we use Chain of Thought  prompting to generate classification and rationale. 
```
python dataset-prep.py
```

Fine-tune the smaller model locally or on Vertex AI workbench. 
```
# AIP_MODEL_DIR="/out" for local
export AIP_MODEL_DIR="/out"
python train.py
```
This will output the model in the "/out" directory.


To test the trained model locally or Vertex AI workbench update the mode path in the test.py script.

```
python test.py
```


To fine tune the smaller model on Vertex AI.
1. Build Custom Training Image 
2. Update custom_trainer.py with your new image name and tag.
3. Run python custom_trainer.py
4. Ensure cloud storage bucket is in same region as you job location

```
# Login to GCP
gcloud auth application-default login
# Update project and reposiotry  name 
PROJECT_REPO_NAME=<yourproject/reponame>
# Build custom container using Cloud Builds.
gcloud builds submit --tag europe-west4-docker.pkg.dev/$PROJECT_REPO_NAME/flan_t5_base_finetune
```
Launch training on Vertext AI.
``` 
python custom_trainer.py
```
Download the model from the cloud storage  to your workbench or locally for testing.

