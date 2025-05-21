# Team Collaboration with DVC

> [!important]
>
> Never include sensitive information such as `gdrive_client_id` or `gdrive_client_secret` in `.dvc/config`**. Instead, store them in** `.dvc/config.local` **or use environment variables for secure management.**

## 🧩 Components of Team Collaboration

### 🌐 1. Git Repository (Code, Pipelines, Parameters, Metadata)

Shared content:

- `.dvc` files (tracking data files)
- `dvc.yaml` / `dvc.lock` (pipeline configuration)
- `params.yaml` (parameters)
- `metrics.json` (metrics)
- Code (model training, preprocessing, etc.)

Pushed to GitHub/GitLab for team members to clone.

------

### ☁️ 2. DVC Remote Storage (Shared Data and Model Content)

Shared content:

- Data files (large files, models) are not stored in Git but uploaded to DVC remote storage.

Team members can download these using `dvc pull`.

------

## ✅ Recommended Team Collaboration Workflow

### 👩‍💻 A. Initial Setup (Completed by the <u>Project Lead</u>)

1. **Configure DVC Remote Storage** (local path, S3, GDrive, SSH, etc.):

   ```bash
   dvc remote add -d storage gdrive://folder-id
   dvc remote modify storage gdrive_client_id xxx --local
   dvc remote modify storage gdrive_client_secret xxx --local
   ```

2. **Create `.dvc` Files or `dvc.yaml` to Manage Data/Pipelines**:

   ```bash
   dvc add data.csv
   dvc run -n train -d train.py -d data.csv -o model.pkl python train.py
   git add data.csv.dvc dvc.yaml dvc.lock
   git commit -m "Add data and training stage"
   ```

3. **Push Data and Git Content**:

   ```bash
   dvc push
   git push
   ```

------

### 👥 B. Collaboration Workflow for <u>Other Team Members</u>

1. **Clone the Repository from GitHub**:

   ```bash
   git clone https://github.com/your-team/project.git
   cd project
   ```

2. **Configure Remote Access Credentials** (using `.dvc/config.local` or environment variables):

   ```bash
   dvc remote modify storage gdrive_client_id xxx --local
   dvc remote modify storage gdrive_client_secret xxx --local
   ```

3. **Pull Data and Models**:

   ```bash
   dvc pull
   ```

4. **Run Pipelines or Continue Development**:

   ```bash
   dvc repro
   ```

5. **Push Changes After Modifications**:

   ```bash
   git commit -am "Update model parameters"
   dvc push
   git push
   ```

------

## 🛡 Security Recommendations (Team Collaboration)

| Recommendation                                    | Description                                                  |
| ------------------------------------------------- | ------------------------------------------------------------ |
| Do not commit `.dvc/config.local`                 | It contains sensitive keys and should be ignored by `.gitignore`. |
| Use environment variables for storing keys        | Suitable for CI/CD or shared servers.                        |
| Use a shared GDrive account or S3 IAM role        | Simplifies permission management.                            |
| Add `.dvc/tmp/` and `.dvc/cache/` to `.gitignore` | Prevents accidental commits of local cache.                  |
