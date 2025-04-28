

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time
import logging
import re
from urllib.parse import urlparse
import random
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleScraper:
    def __init__(self, timeout=20, max_retries=2, use_js=True):
        """Initialize the ArticleScraper with optimized parameters for dynamic content."""
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_js = use_js
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self.driver = None

        if use_js:
            self._init_selenium()

    def _init_selenium(self):
        """Initialize Selenium with anti-detection measures."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')
            
            # Anti-bot detection measures
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option("useAutomationExtension", False)

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Further anti-detection
            self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                """
            })
            
            self.driver.set_page_load_timeout(self.timeout)
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {str(e)}")
            self.use_js = False

    def _get_html_with_selenium(self, url):
        """Get HTML using Selenium with progressive loading techniques."""
        if not self.driver:
            return None

        try:
            # Clean session state
            self.driver.delete_all_cookies()

            # Load the page
            self.driver.get(url)
            
            # Initial wait for basic content
            time.sleep(3)
            
            # Handle consent popups common on news sites
            try:
                consent_buttons = self.driver.find_elements(By.XPATH, 
                    "//*[contains(text(), 'Accept') or contains(text(), 'agree') or contains(text(), 'Consent') or contains(text(), 'Accept all')]")
                for button in consent_buttons:
                    if button.is_displayed():
                        button.click()
                        time.sleep(1)
                    break
            except:
                pass
            
            # Progressive scrolling - mimic human behavior
            scroll_pause_time = 1
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            for i in range(5):  # Scroll in stages
                # Scroll down in steps
                current_scroll = (i+1) * last_height / 5
                self.driver.execute_script(f"window.scrollTo(0, {current_scroll});")
                
                # Random pause like a human
                time.sleep(scroll_pause_time + random.uniform(0.5, 1.5))
                
                # Check if we need to handle any lazy-loaded iframes
                iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
                if iframes and len(iframes) > 0:
                    # Log iframe discovery for debugging
                    logger.info(f"Found {len(iframes)} iframes, attempting to process...")
                    iframe_content = self._check_iframes()
                    if iframe_content:
                        return iframe_content
            
            # Check for JSON-LD structured data in scripts
            json_ld = self._extract_json_ld()
            if json_ld:
                return json_ld
                
            # Return the fully rendered page
            return self.driver.page_source

        except Exception as e:
            logger.error(f"Selenium error: {str(e)}")
            return None
            
    def _check_iframes(self):
        """Check iframes for content."""
        try:
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            
            for iframe in iframes:
                try:
                    # Switch to iframe
                    self.driver.switch_to.frame(iframe)
                    time.sleep(1)
                    
                    # Look for article content
                    article_content = self.driver.find_elements(By.CSS_SELECTOR, 
                        "article, [class*='article'], [class*='content'], [class*='story']")
                    
                    if article_content:
                        html = self.driver.page_source
                        # Switch back to main content
                        self.driver.switch_to.default_content()
                        return html
                        
                    # Switch back to main content
                    self.driver.switch_to.default_content()
                except:
                    # Ensure we switch back if there's an error
                    self.driver.switch_to.default_content()
                    continue
            
            return None
        except:
            # Ensure we switch back to main content
            try:
                self.driver.switch_to.default_content()
            except:
                pass
            return None

    def _extract_json_ld(self):
        """Extract article content from JSON-LD."""
        try:
            scripts = self.driver.find_elements(By.CSS_SELECTOR, "script[type='application/ld+json']")
            
            for script in scripts:
                try:
                    json_text = script.get_attribute('textContent')
                    data = json.loads(json_text)
                    
                    # Check various JSON-LD structures
                    if isinstance(data, dict):
                        # Article schema
                        if data.get('@type') == 'Article' or data.get('@type') == 'NewsArticle':
                            if 'articleBody' in data:
                                return f"<html><body><article>{data['articleBody']}</article></body></html>"
                    
                    # Handle array of schemas
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and (item.get('@type') == 'Article' or item.get('@type') == 'NewsArticle'):
                                if 'articleBody' in item:
                                    return f"<html><body><article>{item['articleBody']}</article></body></html>"
                except:
                    continue
                    
            return None
        except:
            return None

    def _get_html(self, url):
        """Get HTML with automatic fallback between methods."""
        # Try Selenium first if enabled
        if self.use_js:
            html = self._get_html_with_selenium(url)
            if html:
                return html

        # Fallback to requests
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed to fetch {url}: {str(e)}")
                    return None
                time.sleep(1)

    def _calculate_text_density(self, element):
        """Calculate text density for determining content-rich areas."""
        text = element.get_text(strip=True)
        html = str(element)
        if not html or len(html) == 0:
            return 0
        return len(text) / len(html)

    def _extract_main_content(self, soup):
        """Extract content using multiple heuristics."""
        # 1. Remove non-content elements
        for element in soup.select('header, footer, nav, aside, [class*="banner"], [class*="ad"], [id*="ad"], [class*="menu"], [class*="comment"], [class*="related"], [class*="share"], script, style, noscript'):
            if element:
                element.decompose()
        
        # 2. Try common article containers
        container_selectors = [
            'article', 'main article', 
            '[class*="article-body"]', '[class*="article-content"]', '[class*="story-content"]',
            '[class*="post-content"]', '[class*="entry-content"]', 
            '[itemprop="articleBody"]', '.content-body', '.story',
            '[data-testid="article-container"]', 
            '[data-testid="content-canvas"]', '.canvasContent',
            '.content', '#content', '.main-content', 'main'
        ]
        
        for selector in container_selectors:
            containers = soup.select(selector)
            if containers:
                # Get container with most paragraph content
                container = max(containers, key=lambda c: len(c.find_all('p'))) if len(containers) > 1 else containers[0]
                paragraphs = container.find_all('p')
                if paragraphs and len(paragraphs) >= 2:
                    content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 25)
                    if content and len(content) > 150:
                        return self._clean_content(content)
        
        # 3. Try finding content-rich divs using text density
        divs = soup.find_all('div')
        content_divs = []
        
        for div in divs:
            # Skip small or suspicious divs
            if len(str(div)) < 500:
                continue
                
            if div.get('class') and any(cls in ['sidebar', 'widget', 'menu', 'nav', 'header', 'footer'] 
                                        for cls in div.get('class') if cls):
                    continue

            paragraphs = div.find_all('p')
            if len(paragraphs) >= 2:
                text_length = sum(len(p.get_text(strip=True)) for p in paragraphs)
                density = self._calculate_text_density(div)
                
                # Weight by number of paragraphs, text length and density
                score = len(paragraphs) * 10 + text_length / 100 + density * 100
                content_divs.append((div, score))
        
        # Sort by content richness score
        if content_divs:
            content_divs.sort(key=lambda x: x[1], reverse=True)
            best_div = content_divs[0][0]
            
            paragraphs = best_div.find_all('p')
            content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 25)
            if content and len(content) > 150:
                return self._clean_content(content)
        
        # 4. Try JSON-LD extraction (structured data)
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if 'articleBody' in data:
                        return self._clean_content(data['articleBody'])
                    elif '@graph' in data:
                        for item in data['@graph']:
                            if isinstance(item, dict) and 'articleBody' in item:
                                return self._clean_content(item['articleBody'])
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'articleBody' in item:
                            return self._clean_content(item['articleBody'])
            except:
                pass
                
        # 5. Last resort: get all paragraph text
        paragraphs = soup.find_all('p')
        meaningful_paragraphs = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40]
        
        if meaningful_paragraphs:
            content = "\n\n".join(meaningful_paragraphs)
            if len(content) > 150:
                return self._clean_content(content)
        
        return "Unable to extract content from the URL."
    
    def _clean_content(self, content):
        """Clean extracted content."""
        # Remove boilerplate phrases
        content = re.sub(r'Copyright ©.*?(reserved|rights)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'For reprint rights:.*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Follow us on.*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Subscribe to.*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Click here to.*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Advertisement.*', '', content, flags=re.IGNORECASE)
        
        # Form paragraphs from related sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_paragraph.append(sentence)
            
            # Break into paragraphs at natural points
            if len(' '.join(current_paragraph)) > 200 or sentence.endswith(('.', '!', '?')):
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs).strip()

    def c(self, url):
        """Main extraction method with multiple fallback strategies."""
        try:
            # First attempt with standard approach
            html = self._get_html(url)
            if not html:
                return "Unable to extract content from the URL."

            soup = BeautifulSoup(html, "html.parser")
            content = self._extract_main_content(soup)

            # If content extraction failed, try enhanced approaches
            if len(content) < 150 or content == "Unable to extract content from the URL.":
                logger.info("Initial extraction failed, trying enhanced approaches...")
                if self.use_js and self.driver:
                    try:
                        self.driver.set_page_load_timeout(30)
                        self.driver.get(url)
                        time.sleep(10)
                        for _ in range(5):
                            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(2)
                            self.driver.execute_script("window.scrollTo(0, 0);")
                            time.sleep(1)
                        json_content = self._extract_json_ld()
                        if json_content:
                            soup = BeautifulSoup(json_content, "html.parser")
                            content = self._extract_main_content(soup)
                            if len(content) > 150 and content != "Unable to extract content from the URL.":
                                return content
                        iframe_html = self._check_iframes()
                        if iframe_html:
                            iframe_soup = BeautifulSoup(iframe_html, "html.parser")
                            iframe_content = self._extract_main_content(iframe_soup)
                            if len(iframe_content) > 150 and iframe_content != "Unable to extract content from the URL.":
                                return iframe_content
                        html = self.driver.page_source
                        soup = BeautifulSoup(html, "html.parser")
                        content = self._extract_main_content(soup)
                    except Exception as e:
                        logger.error(f"Enhanced extraction error: {str(e)}")

            # Final validation
            if content == "Unable to extract content from the URL." or len(content) < 100:
                logger.warning(f"Failed to extract content from {url}")

            return content

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return "Unable to extract content from the URL."

    def __del__(self):
        """Clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

# Create a single instance for reuse
_scraper = ArticleScraper(use_js=True)

# Legacy interface for backward compatibility
def c(url):
    """Extract content from URL."""
    return _scraper.c(url)

