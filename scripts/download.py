import requests
import os
import tarfile
from bs4 import BeautifulSoup
import re


OPENBSD_VER = "7.7"
MIRROR = "https://cdn.openbsd.org"
ARCH = "amd64"
MAN_TGZ_URL = f"{MIRROR}/pub/OpenBSD/{OPENBSD_VER}/{ARCH}/man{OPENBSD_VER.replace('.','')}.tgz"
DOWNLOAD_DIR = os.path.join(os.path.abspath(os.getcwd()), 'openbsd-doc-raw')
MAN_TGZ_PATH = os.path.join(DOWNLOAD_DIR, f"man{OPENBSD_VER.replace('.', '')}.tgz")
MAN_EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, f"man{OPENBSD_VER.replace('.', '')}")

FAQ_BASE_URL = "https://www.openbsd.org/faq"
FAQ_INDEX_URL = f"{FAQ_BASE_URL}/index.html"
FAQ_DOWNLOAD_DIR = os.path.join(DOWNLOAD_DIR, 'faq')

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FAQ_DOWNLOAD_DIR, exist_ok=True)

print("Trying to download man pages...")
print(f"OpenBSD mirror: {MIRROR}")
print(f"OpenBSD version: {OPENBSD_VER}")
print(f"OpenBSD architecture: {ARCH}")

try:
    response = requests.get(MAN_TGZ_URL, stream=True)
    response.raise_for_status()

    with open(MAN_TGZ_PATH, mode='wb') as file:
        for chunk in response.iter_content(chunk_size=100*1024):
            file.write(chunk)
    print(f"Downloaded the archive successfully to {DOWNLOAD_DIR}")

    print(f"Extracting {MAN_TGZ_PATH} to {MAN_EXTRACT_DIR}...")
    os.makedirs(MAN_EXTRACT_DIR, exist_ok=True)
    with tarfile.open(MAN_TGZ_PATH, "r:gz") as tar:
        tar.extractall(path=MAN_EXTRACT_DIR)
    print("Extraction complete.")

except requests.exceptions.RequestException as e:
    print(f"Error downloading man.tgz: {e}")
except tarfile.ReadError as e:
    print(f"Error extracting man.tgz: {e}")

print("Trying to download the FAQ")
print(f"FAQ URL: {FAQ_BASE_URL}")
print(f"Downloading FAQ from {FAQ_INDEX_URL}")

try:
    response = requests.get(FAQ_INDEX_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    faq_links = set()

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Basic regex to match faqN.html, excluding index.html itself or external links
        if re.match(r'^faq\d+\.html$', href) or href == "index.html":
            full_url = requests.compat.urljoin(FAQ_INDEX_URL, href)
            faq_links.add(full_url)

    print(f"Found {len(faq_links)} FAQ sections. Downloading...")
    for link in sorted(list(faq_links)): # Sort for consistent order
        filename = os.path.basename(link)
        filepath = os.path.join(FAQ_DOWNLOAD_DIR, filename)
        print(f"Downloading {link} to {filepath}...")
        faq_response = requests.get(link)
        faq_response.raise_for_status()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(faq_response.text)
        print(f"Downloaded {filename}")

except requests.exceptions.RequestException as e:
    print(f"Error downloading FAQs: {e}")
