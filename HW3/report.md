# Practice 3 Report

**GitHub Repository:** https://github.com/Artem-Lishchuk/MLOps  
**Remote Storage:** Backblaze B2 (S3-compatible)

## 1. Remote Storage Setup (Backblaze B2)

A new private bucket named `ml-ops-lishchuk` was created in Backblaze B2. The bucket is configured to keep all versions of files, providing a robust history for DVC-tracked data.

**Bucket Details:**
- **Name:** `ml-ops-lishchuk`
- **Endpoint:** `s3.us-east-005.backblazeb2.com`
- **Type:** Private

![Backblaze Bucket](assets/bucket.png)

To allow DVC to communicate with the bucket, an **Application Key** was generated with appropriate read/write permissions.

![Application Key](assets/application_key.png)

---

## 2. DVC Configuration

The DVC remote was updated to use the S3-compatible API provided by Backblaze.

### Terminal Commands:

```powershell
# Set the S3 endpoint URL for Backblaze
dvc remote modify dvstore endpointurl https://s3.us-east-005.backblazeb2.com

# Configure credentials locally (not committed to Git)
dvc remote modify dvstore access_key_id 'KEY_ID' --local
dvc remote modify dvstore secret_access_key 'KEY' --local
```

### Configuration Verification:

The `dvc remote list` command confirms that `dvstore` is now the default remote pointing to the S3 bucket.

![DVC Remotes](assets/remotes.png)

---