from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="resources", repo_id="AiphoriaTech/T-one", repo_type="model")
