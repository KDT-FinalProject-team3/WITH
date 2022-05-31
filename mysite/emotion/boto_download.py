from datetime import datetime
import boto3


ACCESS_KEY_ID = 'AKIA5VZTIAOJRRBQJPIK' #s3 관련 권한을 가진 IAM계정 정보
ACCESS_SECRET_KEY = '+eQg3dpuoVAWWjrXXtAsYBeIDBnNxpPYpp9M+yuq'
BUCKET_NAME = 'msa-test03'

#start = datetime.now()
#fname_sound = start.strftime('%Y%m%d_%H%M%S')

file_name = 'test_dir/test.txt'  # 저장할 파일경로 및 파일명(이 파일명으로 저장됨(test.txt다른 형식으로 지정가능))
key = 'iot-test/test.txt'

# 조건문 걸어서 날짜양식 정확하면 다운로드
# 아니면 pass , 유효성 검사사
#key = f'iot-test/{fname_sound}' # s3폴더 경로 및 파일 경로

s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=ACCESS_SECRET_KEY)
s3_client.download_file(BUCKET_NAME, key, file_name)




