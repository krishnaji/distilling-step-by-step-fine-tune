from google.cloud import aiplatform

LOCATION = 'asia-southeast1'
BUCKET = "gs://flant5finetune"
PROJECT_ID = "genai-380800"

aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)

job = aiplatform.CustomContainerTrainingJob(
        display_name="flan-t5-base-fine-tune",
        container_uri="europe-west4-docker.pkg.dev/genai-380800/genai/flan_t5_base_finetune:metrics",
        model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest",
    )

model = job.run(
   model_display_name="flan-t5-base-fine-tune",
   replica_count=1,
    machine_type="a2-highgpu-8g",
    accelerator_type="NVIDIA_TESLA_A100",
    boot_disk_size_gb=5024,
    accelerator_count = 8,
)
