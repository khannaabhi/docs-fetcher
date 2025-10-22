import os
import time
import requests
import html2text
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def scrape_documentation(base_url, output_dir, path_filter):
    """
    Scrapes the main content of a documentation website, converts it to Markdown,
    and saves the files locally, following only links that match a specific path.

    Args:
        base_url (str): The starting URL of the documentation.
        output_dir (str): The directory to save the scraped Markdown files.
        path_filter (str): A string that must be in the URL path to be followed (e.g., '/docs/').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created directory: {output_dir}")

    visited_urls = set()
    urls_to_visit = [base_url]

    markdown_converter = html2text.HTML2Text()
    markdown_converter.body_width = 0 # Don't wrap lines

    while urls_to_visit:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        print(f"Scraping: {current_url}")
        time.sleep(1) 

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ Error fetching {current_url}: {e}")
            continue

        visited_urls.add(current_url)

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- 1. Robust Content Area Identification ---
        # Prioritize semantic HTML5 elements for main content.
        # Then, look for common div classes that hold main content.
        content_area_selectors = [
            'main',                   # Standard semantic HTML for main content
            'article',                # For self-contained content
            'div.docs-content',       # Common for documentation sites
            'div.main-content',       # Another common pattern
            'div.content-wrapper',    # General content container
            'div.td-main',            # As seen in Istio, useful general if not too specific
            'div#content',            # Common ID for content
            'div[role="main"]',       # ARIA role for main content
        ]
        content_area = None
        for selector in content_area_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                break
        
        if not content_area:
            print(f"⚠️ Could not find a primary content area for {current_url}. Falling back to body content.")
            # Fallback to body content if no specific content area found,
            # but this increases the risk of including unwanted elements.
            content_area = soup.find('body')
            if not content_area: # Should ideally not happen for valid HTML
                print(f"❌ Could not even find body for {current_url}. Skipping.")
                continue

        # --- 2. Generalized Sidebar Removal ---
        # Strategy:
        # a. Target common semantic navigation tags (nav, aside).
        # b. Target elements with common "sidebar-like" class/ID names (e.g., "sidebar", "nav", "menu").
        # c. Crucially, *exclude* elements that are direct ancestors or within the identified `content_area`.
        
        removed_sidebars_count = 0

        # Define a broad set of selectors for potential sidebars.
        # Order of selectors might matter for performance, but for `decompose`, not functionally.
        potential_sidebar_selectors = [
            'nav',                    # Most common semantic tag for navigation
            'aside',                  # Semantic tag for content indirectly related to main content
            '[role="navigation"]',    # ARIA role for navigation
            '[class*="sidebar"]',     # Classes containing "sidebar" (e.g., "left-sidebar", "sidebar-menu")
            '[id*="sidebar"]',        # IDs containing "sidebar"
            '[class*="sidenav"]',     # Classes containing "sidenav"
            '[id*="sidenav"]',        # IDs containing "sidenav"
            '[class*="nav-menu"]',    # Classes containing "nav-menu"
            '[id*="nav-menu"]',       # IDs containing "nav-menu"
            '[class*="table-of-contents"]', # Some TOCs are standalone sidebars
            '[id*="toc"]',            # Common ID for Table of Contents
            # Add other common patterns you observe across documentation sites:
            # 'div.td-sidebar-nav',   # Specific to Istio, but if you want to include common ones
            # '.md-sidebar',          # Common in MkDocs-based themes
            # '.doc-sidebar',
        ]

        # Get all potential sidebar elements
        all_potential_sidebars = []
        for selector in potential_sidebar_selectors:
            all_potential_sidebars.extend(soup.select(selector))
        
        # Use a set to avoid processing the same element multiple times if it matches multiple selectors
        unique_potential_sidebars = set(all_potential_sidebars)

        for element_to_remove in unique_potential_sidebars:
            # IMPORTANT: Ensure we are not removing the main content area itself,
            # or any of its direct children or crucial ancestors.
            # We want to remove elements *around* the content_area.

            # Check if the element to remove is the content_area itself
            if element_to_remove == content_area:
                continue

            # Check if the element to remove contains the content_area
            # (i.e., content_area is a descendant of element_to_remove)
            if content_area in element_to_remove.find_all(True): # find_all(True) gets all descendants
                # If content_area is inside this element, this element might be a legitimate parent wrapper
                # that we don't want to remove entirely, or it's a very broad selector.
                # In such cases, we might need to be more surgical or skip this element.
                # For now, let's skip removing an element that *contains* the main content.
                continue

            # Check if the element to remove is an ancestor of the content_area
            # This is implicitly covered by the previous check if content_area is a descendant.
            # A more direct check: if content_area.find_parent(element_to_remove) is not None

            # Check if the element to remove is one of the parents of the main content area
            # that we've already identified as a content area (e.g., 'div.main-content').
            # This prevents us from removing the very container we identified as content.
            is_ancestor_of_content_area = False
            current = content_area
            while current:
                if current == element_to_remove:
                    is_ancestor_of_content_area = True
                    break
                current = current.parent
            if is_ancestor_of_content_area:
                continue
            
            # Additional heuristic: avoid removing elements that are direct parents of <body>,
            # which could be major layout wrappers, unless they explicitly look like a sidebar.
            # This is tricky as some sites put sidebars directly under <body>.
            # A safer approach is to ensure the element isn't broadly structuring the page.
            # For now, rely on the content_area check.

            # Perform removal
            try:
                element_to_remove.decompose()
                removed_sidebars_count += 1
                # print(f"  -> ✔️ Removed element matching selector: {element_to_remove.name} (class: {element_to_remove.get('class')})")
            except Exception as e:
                print(f"  -> ❌ Error decomposing element: {element_to_remove.name} - {e}")

        if removed_sidebars_count > 0:
            print(f"  -> ✔️ Removed {removed_sidebars_count} potential sidebar/navigation elements.")
        else:
            print("  -> ℹ️ No common sidebar/navigation elements found or removed based on general selectors.")

        # --- 3. Convert to Markdown and Save ---
        # Ensure content_area is still valid after potential decompositions
        if not content_area:
            print(f"❌ Content area disappeared after sidebar removal for {current_url}. Skipping content save.")
            continue

        html_content = str(content_area)
        markdown_content = markdown_converter.handle(html_content)

        parsed_url = urlparse(current_url)
        path_segments = parsed_url.path.strip('/').split('/')
        
        filename = "index.md" if not path_segments[-1] else path_segments[-1] + ".md"
        
        dir_path = os.path.join(output_dir, *path_segments[:-1])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = os.path.join(dir_path, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"  -> ✅ Saved to {file_path}")
        except OSError as e:
            print(f"❌ Error saving file {file_path}: {e}")

        # --- 4. Discover New Links ---
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            if href.startswith('#'):
                continue

            absolute_url = urljoin(current_url, href).split('#')[0]

            if (absolute_url not in visited_urls and
                urlparse(absolute_url).netloc == urlparse(base_url).netloc and
                path_filter in absolute_url):
                
                urls_to_visit.append(absolute_url)

    print("\nScraping complete! ✨")


# if __name__ == '__main__':
#     # --- Example 1: Pydantic Docs ---
#     print("\n--- Scraping Pydantic Docs ---")
#     PYDANTIC_BASE_URL = 'https://docs.pydantic.dev/latest/'
#     PYDANTIC_OUTPUT_DIR = 'pydantic_docs_general'
#     PYDANTIC_PATH_FILTER = '/latest/'
#     scrape_documentation(
#         base_url=PYDANTIC_BASE_URL,
#         output_dir=PYDANTIC_OUTPUT_DIR,
#         path_filter=PYDANTIC_PATH_FILTER
#     )

#     # --- Example 2: Istio Docs ---
#     # Will use the generalized removal, not Istio specific selectors.
#     print("\n--- Scraping Istio Docs (with general removal) ---")
#     ISTIO_BASE_URL = 'https://istio.io/latest/docs/'
#     ISTIO_OUTPUT_DIR = 'istio_docs_general'
#     ISTIO_PATH_FILTER = '/docs/'
#     scrape_documentation(
#         base_url=ISTIO_BASE_URL,
#         output_dir=ISTIO_OUTPUT_DIR,
#         path_filter=ISTIO_PATH_FILTER
#     )
    
#     # --- Example 3: MkDocs Material Theme (common for docs) ---
#     # This often uses a sidebar.
#     print("\n--- Scraping MkDocs Material Theme Example (with general removal) ---")
#     MKDOCS_BASE_URL = 'https://squidfunk.github.io/mkdocs-material/getting-started/'
#     MKDOCS_OUTPUT_DIR = 'mkdocs_material_docs_general'
#     MKDOCS_PATH_FILTER = '/mkdocs-material/' # Adjust path filter if crawling deeper
#     scrape_documentation(
#         base_url=MKDOCS_BASE_URL,
#         output_dir=MKDOCS_OUTPUT_DIR,
#         path_filter=MKDOCS_PATH_FILTER
#     )

#     # --- Example 4: Read the Docs (Sphinx documentation) ---
#     # Another very common documentation platform
#     print("\n--- Scraping Read the Docs Example (with general removal) ---")
#     READTHEDOCS_BASE_URL = 'https://www.sphinx-doc.org/en/master/usage/quickstart.html'
#     READTHEDOCS_OUTPUT_DIR = 'sphinx_docs_general'
#     READTHEDOCS_PATH_FILTER = '/en/master/' # Adjust path filter
#     scrape_documentation(
#         base_url=READTHEDOCS_BASE_URL,
#         output_dir=READTHEDOCS_OUTPUT_DIR,
#         path_filter=READTHEDOCS_PATH_FILTER
#     )