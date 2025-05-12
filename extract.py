import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm


class INaturalistScraper:
    """
    A class to scrape images from iNaturalist for a given taxon URL.
    """
    def __init__(self, url, label, max_images=200):
        self.url = url
        self.label = label.replace(" ", "_")
        self.max_images = max_images  # Maximum number of images to download
        self.setup_driver()
        self.setup_directories()

    def setup_directories(self):
        """
        Create a directory for the label if it doesn't exist.
        """
        self.data_dir = os.path.join("training_data", self.label)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_driver(self):
        """
        Set up the Selenium WebDriver with Chrome options.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)

    def get_image_urls(self):
        """
        Scrape image URLs from the iNaturalist page.
        """
        print(f"Accessing {self.url}")
        self.driver.get(self.url)

        try:
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "TaxonPhoto")))
        except TimeoutException:
            print("Timeout waiting for page to load")
            return []

        print("Loading images...")
        image_urls = set()
        scroll_attempts = 0
        max_scroll_attempts = 10  # Limit scrolling to avoid endless loops

        with tqdm(total=self.max_images, desc="Finding images") as pbar:
            while len(image_urls) < self.max_images and scroll_attempts < max_scroll_attempts:
                # Find all image elements using multiple selectors
                selectors = [
                    "//div[contains(@class, 'TaxonPhoto')]//div[contains(@class, 'CoverImage')]",
                    "//div[contains(@class, 'PhotoModal--photo')]//img",
                    "//div[contains(@class, 'TaxonPhoto')]//img"
                ]

                for selector in selectors:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if len(image_urls) >= self.max_images:
                            break

                        try:
                            style = element.get_attribute('style')
                            if style and 'background-image' in style:
                                url = style.split('url("')[1].split('")')[0]
                                if url and url.startswith('http'):
                                    image_urls.add(url)
                                    pbar.update(1)

                            src = element.get_attribute('src')
                            if src and src.startswith('http'):
                                image_urls.add(src)
                                pbar.update(1)

                        except:
                            continue

                    if len(image_urls) >= self.max_images:
                        break

                last_height = self.driver.execute_script("return document.body.scrollHeight")
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = self.driver.execute_script("return document.body.scrollHeight")

                if new_height == last_height:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0

        return list(image_urls)[:self.max_images]

    def download_images(self, image_urls):
        """
        Download images from the provided URLs and save them to the data directory.
        """
        print(f"\nDownloading {len(image_urls)} images...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        with tqdm(total=len(image_urls), desc="Downloading") as pbar:
            for idx, url in enumerate(image_urls, 1):
                try:
                    url = url.replace('&quot;', '').strip()
                    if not url.startswith('http'):
                        continue

                    filename = f"{self.label}_{idx}.jpg"
                    filepath = os.path.join(self.data_dir, filename)

                    if os.path.exists(filepath):
                        print(f"\nSkipping {filename} (already exists)")
                        pbar.update(1)
                        continue

                    response = requests.get(url, headers=headers)
                    response.raise_for_status()

                    with open(filepath, 'wb') as f:
                        f.write(response.content)

                    pbar.update(1)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"\nError downloading image {idx}: {str(e)}")

    def run(self):
        """
        Run the scraper to download images from the iNaturalist page.
        """
        try:
            print(f"\nStarting scraping for {self.label}")
            image_urls = self.get_image_urls()

            if not image_urls:
                print("No images found!")
                return

            print(f"\nFound {len(image_urls)} unique images")
            self.download_images(image_urls)

        finally:
            self.driver.quit()
            print("\nScraping completed!")

def main():
    """
    Main function to run the iNaturalist scraper.
    """
    # replace with the desired iNaturalist URL and label
    url = "https://www.inaturalist.org/taxa/202005-Iridomyrmex-anceps/browse_photos"
    label = "Iridomyrmex anceps"
    max_images = 100  # Set maximum number of images to download

    scraper = INaturalistScraper(url, label, max_images)
    scraper.run()

if __name__ == "__main__":
    main()
