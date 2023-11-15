#%%
import os
import requests
import gzip
import shutil
import io
import zipfile
import tarfile
import urllib.request

def download_file_from_url(url, folder, filename):
    # フォルダが存在しない場合は作成します
    os.makedirs(folder, exist_ok=True)

    # ファイルをダウンロードします
    response = requests.get(url)

    # レスポンスのステータスコードが正常でない場合は例外を送出します
    response.raise_for_status()

    # ダウンロードしたファイルを指定のフォルダに保存します
    with open(os.path.join(folder, filename), "wb") as f:
        f.write(response.content)


def download_and_extract_gz(url: str, output_dir: str, filename) -> str:
    """
    URLから.gzファイルをダウンロードし、解凍して保存する関数。

    Args:
        url (str): ダウンロードするファイルのURL。
        output_dir (str): 解凍して保存するディレクトリのパス。

    Returns:
        str: 解凍して保存されたファイルのパス。
    """

    # ダウンロードしたファイルを保存するパスを作成
    output_file = os.path.join(output_dir, filename)

    # URLからファイルをダウンロード
    response = requests.get(url, stream=True)

    # gzipで解凍してファイルに保存
    with gzip.GzipFile(fileobj=response.raw, mode='rb') as gzip_file, open(output_file, 'wb') as fout:
        shutil.copyfileobj(gzip_file, fout)

    return output_file



def download_and_extract_zip(url: str, output_dir: str = './', extract_dir: str = 'extracted') -> str:
    """
    URLからzipファイルをダウンロードし、解凍して新しいディレクトリに保存する関数。

    Args:
        url (str): ダウンロードするファイルのURL。
        output_dir (str): 解凍して保存するディレクトリの親ディレクトリのパス。
        extract_dir (str): 解凍後のファイルを保存するディレクトリの名前。

    Returns:
        str: 解凍して保存されたファイルのパス。
    """
    # ファイル名を取得
    filename = url.split('/')[-1]

    # ダウンロードしたファイルを保存するパスを作成
    output_file = os.path.join(output_dir, filename)

    # URLからファイルをダウンロード
    response = requests.get(url)

    # zipファイルを解凍してファイルに保存
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        # 解凍先のディレクトリを作成
        extract_path = os.path.join(output_dir, extract_dir)
        os.makedirs(extract_path, exist_ok=True)
        # zipファイル内のファイルを解凍して保存
        zip_file.extractall(extract_path)

    # 解凍後のファイルのパスを返す
    return extract_path

def download_and_extract_tar_gz(url, save_dir, file_name):
    """Download and extract a .tar.gz file from a given url and save it in the specified directory with a custom name.

    Args:
        url (str): URL of the .tar.gz file to download.
        save_dir (str): Path of the directory to save the extracted files to.
        file_name (str): Name of the file to save the downloaded .tar.gz file as.
    """
    # Download the file
    urllib.request.urlretrieve(url, file_name)

    # Extract the files
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=save_dir)

    # Remove the downloaded file
    os.remove(file_name)
        
#%%
if __name__ == "__main__":
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    folder = '/home1/data/common-data/COCO/'
    dir_name = 'annotations'
    os.makedirs(folder+dir_name)
    
    download_and_extract_zip(url, folder, dir_name)# %%

# %%
