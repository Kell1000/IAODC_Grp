import os
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup

nb_pages = 5

# Folder where all PDFs will be saved
DOWNLOAD_FOLDER = "pdf_files"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})


def download_file(pdf_url, filename):
    filepath = os.path.join(DOWNLOAD_FOLDER, filename)

    response = session.get(pdf_url, timeout=60)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath


for n_page in range(1, nb_pages + 1):

    url = f"https://core.ac.uk/search?q=machine+learning&page={n_page}"

    print(f"\nHTTP GET: {url}")

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find links containing PDFs
        all_urls = soup.find_all("a", href=True)

        for link in all_urls:

            href = link["href"]

            if ".pdf" not in href.lower():
                continue

            # Convert relative URL to absolute URL
            pdf_url = urllib.parse.urljoin(url, href)

            print(f"Downloading: {pdf_url}")

            try:
                pdf_response = session.get(
                    pdf_url,
                    timeout=60,
                    allow_redirects=True
                )
                pdf_response.raise_for_status()

                # Get filename from final URL
                parsed_url = urllib.parse.urlparse(pdf_response.url)
                filename = os.path.basename(parsed_url.path)

                # If URL doesn't contain a filename
                if not filename or not filename.lower().endswith(".pdf"):
                    filename = f"paper_page_{n_page}.pdf"

                # Remove unsafe characters
                filename = re.sub(
                    r'[<>:"/\\|?*]',
                    "_",
                    filename
                )

                # Prevent overwriting existing files
                filepath = os.path.join(DOWNLOAD_FOLDER, filename)

                if os.path.exists(filepath):
                    name, extension = os.path.splitext(filename)
                    counter = 1

                    while os.path.exists(filepath):
                        new_filename = f"{name}_{counter}{extension}"
                        filepath = os.path.join(
                            DOWNLOAD_FOLDER,
                            new_filename
                        )
                        counter += 1

                    filename = new_filename

                # Save PDF
                filepath = download_file(pdf_response.url, filename)

                print(f"Saved: {filepath}")

            except requests.RequestException as e:
                print(f"PDF download failed: {e}")

    except requests.RequestException as e:
        print(f"Page request failed: {e}")


print("\nDone!")
print(f"PDFs are saved in: {os.path.abspath(DOWNLOAD_FOLDER)}")
