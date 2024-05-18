from pydrive.auth import GoogleAuth

drive_auth = GoogleAuth()

drive_auth.settings[‘client_config_file’] = r’example\client_secrets.json’

drive_auth.LocalWebserverAuth()
