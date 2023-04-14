# upload saved model to Azure storage blob

import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pickle
from io import StringIO, BytesIO
import pandas as pd

# application id e96200c8-8ee3-420b-ba9e-721c7ed643bf
# object id c6c3b79d-8c28-418e-8a70-785b579010c4
# directory (tenent) id 7bf3642b-7ac1-48b3-af25-6ca1f0425788

# secret value Zov8Q~XhtyXKJhj.JM~unJsig2GntpI8Hrf15cBt
# secret ID ccff1bd5-0c29-402a-be64-79beb5a7bcaf

#  KEY_VAULT_NAME=local_set_vault_key_article-rec

class blobConn:

    def __init__(self):
        self.account_url = "https://cs4100320028135ba5f.blob.core.windows.net"
        self.default_credential = DefaultAzureCredential()
        
        # Create the BlobServiceClient object
        self.blob_service_client = BlobServiceClient(self.account_url, credential=self.default_credential)
        #self.container_client = self.blob_service_client.create_container(self.container_name)
        #blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

    def create_container(self, container_name):
        container_client = self.blob_service_client.create_container(container_name)
        #return container_client


    def upload(self,upload_file_path, container_name, blob_name):
        container_client = self.blob_service_client.get_container_client(container= container_name)
        blob_client = container_client.get_blob_client(blob_name)
        # Upload the created file
        with open(file=upload_file_path, mode="rb") as data:
            blob_client.upload_blob(data)

    def download_local(self, download_file_path, container_name, blob_name):
        container_client = self.blob_service_client.get_container_client(container= container_name)
        #blob_client = container_client.get_blob_client(blob_name)
 
        # download the file
        with open(file=download_file_path, mode="wb") as download_file:
            download_file.write(container_client.download_blob(blob_name).readall())

    def download(self, container_name, blob_name, file_type:str):
        blob_client = self.blob_service_client.get_blob_client(container=container_name,blob=blob_name)
        if file_type == 'pickle':
            blob_data = blob_client.download_blob().readall()
            ar = pickle.loads(blob_data)
        elif file_type =='csv':
            blobstring = blob_client.download_blob()
            ar = pd.read_csv(StringIO(blobstring.content_as_text()))
        else:
            ar = blob_client.download_blob().readall()
        return ar
    
    #def download(self, container_name, blob_name):
    #    blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    #    blob_data = blob
       

if __name__ == '__main__':
    pkl_path = '/Users/yang_home/Documents/learning/AI_dev/recommender_system/pre_trained/articles_embeddings_test.pickle'
    #print(DefaultAzureCredential())
    #blobConn().create_container('rec-model-v1')
    print('container created')
    blobConn().upload(pkl_path,'rec-model-v1','pre-trained-embeddings')