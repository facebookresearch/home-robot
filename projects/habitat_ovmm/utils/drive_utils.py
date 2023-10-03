import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API credentials
SCOPES = ["https://www.googleapis.com/auth/drive"]
CLIENT_SECRET_FILE = "client_secret.json"  # Update with your file name
API_NAME = "drive"
API_VERSION = "v3"

# Folder ID of the destination folder in Google Drive
DESTINATION_FOLDER_ID = (
    "1Hvy2AEZ9C1_a6AFwqEPoe3K7Q7v5di-K"  # Update with your folder ID
)


def authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_authorized_user_file())
    return creds


def upload_to_drive(service, local_path, parent_id):
    file_name = os.path.basename(local_path)
    media = MediaFileUpload(local_path)

    file_metadata = {"name": file_name, "parents": [parent_id]}

    uploaded_file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )

    print(f'Uploaded {file_name} with ID: {uploaded_file["id"]}')


def upload_folder_contents(service, local_folder, parent_id):
    for item in os.listdir(local_folder):
        item_path = os.path.join(local_folder, item)
        if os.path.isfile(item_path):
            upload_to_drive(service, item_path, parent_id)
        elif os.path.isdir(item_path):
            folder_name = os.path.basename(item_path)
            folder_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            created_folder = (
                service.files().create(body=folder_metadata, fields="id").execute()
            )
            upload_folder_contents(service, item_path, created_folder["id"])


def main():
    creds = authenticate()
    service = build(API_NAME, API_VERSION, credentials=creds)
    upload_folder_contents(service, "video_dir", DESTINATION_FOLDER_ID)


if __name__ == "__main__":
    main()
