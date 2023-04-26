# upload saved model to Azure storage blob

import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pickle
from io import StringIO, BytesIO
import pandas as pd
import numpy as np


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
            '''
        elif file_type == 'npy':
            stream = io.BytesIO()
            blob_service.get_blob_to_stream(container_name, blob_name, stream)
            ar = np.frombuffer(stream.getbuffer())
            '''
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