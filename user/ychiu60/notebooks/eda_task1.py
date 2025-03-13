import re
import os
import xml.etree.ElementTree as ET

import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns

# Define the folder path where the .trec files are stored
folder_path = 'task1-symptom-ranking/erisk25-t1-dataset'

# Get list of .trec files
files = os.listdir(folder_path)
trec_files = [file for file in files if file.endswith(".trec")]

# Store text and post content for further analysis
texts = []
posts = []

# Initialize counters for document statistics
doc_count = 0
pre_lengths = []
text_lengths = []
post_lengths = []

# Iterate over each .trec file and parse it
for trec_file in trec_files:
    file_path = os.path.join(folder_path, trec_file)
    
    # Read and clean the data
    with open(file_path, "r", encoding="utf-8") as file:
        trec_data = file.read()
    
    # Remove problematic & character
    cleaned_data = trec_data.replace("&", "") #.replace("\x1f", "").replace("\x13", "")
    cleaned_data = re.sub(r'[^\x20-\x7E]', '', cleaned_data)
    
    # Wrap the cleaned data with a root tag for XML parsing
    wrapped_data = f"<ROOT>{cleaned_data}</ROOT>"
    
    # Parse the XML
    try:
        root = ET.fromstring(wrapped_data)
        
        # Extract data for each document
        for doc in root.findall("DOC"):
            doc_no = doc.find("DOCNO").text
            pre = doc.find("PRE").text or ""
            text = doc.find("TEXT").text or ""
            post = doc.find("POST").text or ""
            
            # Track document statistics
            doc_count += 1
            pre_lengths.append(len(pre))
            text_lengths.append(len(text))
            post_lengths.append(len(post))
            
            texts.append(text)
            posts.append(post)
    
    except ET.ParseError as e:
        print(f"Error parsing {trec_file}: {e}")

# EDA Analysis

# 1. Number of documents
print(f"Total number of documents: {doc_count}")

# 2. Average text length
avg_text_length = sum(text_lengths) / len(text_lengths)
avg_post_length = sum(post_lengths) / len(post_lengths)

print(f"Average TEXT length: {avg_text_length}")
print(f"Average POST length: {avg_post_length}")

# 3. Text Length Distribution Plot
# sns.histplot(text_lengths, kde=True)
# plt.title('Distribution of TEXT Lengths')
# plt.xlabel('Text Length')
# plt.ylabel('Frequency')
# plt.show()

# sns.histplot(post_lengths, kde=True)
# plt.title('Distribution of POST Lengths')
# plt.xlabel('Post Length')
# plt.ylabel('Frequency')
# plt.show()

# 4. Word Cloud for TEXT field
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texts))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud for TEXT Field")
plt.axis('off')
plt.show()

# 5. Most Common Words in TEXT field
all_text = ' '.join(texts).lower().split()
word_counts = Counter(all_text)
common_words = word_counts.most_common(10)
print(f"Most common words in TEXT: {common_words}")

# 6. Check for missing data (PRE, TEXT, POST)
missing_pre = sum(1 for pre in pre_lengths if pre == 0)
missing_text = sum(1 for text in text_lengths if text == 0)
missing_post = sum(1 for post in post_lengths if post == 0)

print(f"Documents missing PRE: {missing_pre}")
print(f"Documents missing TEXT: {missing_text}")
print(f"Documents missing POST: {missing_post}")



