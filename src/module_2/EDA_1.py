import boto3
import os

# Crear un cliente de S3
s3 = boto3.client('s3')

# Especificar el nombre del bucket y la carpeta de destino
bucket_name = 'zrive-ds-data'
folder_name = 'groceries/sampled-datasets'
local_dir = 'projects/zrive-ds/sampled-datasets/'

# Asegurarse de que el directorio local exista
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# Descargar archivos del bucket
s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(bucket_name)

for obj in bucket.objects.filter(Prefix=folder_name):
    # Definir ruta del archivo local
    local_file_path = os.path.join(local_dir, obj.key.split('/')[-1])
    bucket.download_file(obj.key, local_file_path)
    print(f"Descargado: {local_file_path}")
