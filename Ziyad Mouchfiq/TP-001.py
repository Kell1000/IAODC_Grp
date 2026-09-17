 chiffres Numero: +212 6-61-22-33-44
phone_pattern = r'(?:\+212|0)[\s\-]?[5-7](?:[\s\-]?\d){8}'

# date JJ/MM/AAAA ou AAAA/MM/JJ ou JJ-MM-AAAA ou AAAA-MM-JJ
date_pattern = r'\b(?:\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2}|\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2})\b'

#prix
prix_pattern = r'\b\d+(?:[.,]\d+)?\s*(?:MAD|EUR|USD)\b'

#hastags et mentions
hashtag_pattern = r'#\w+'
mention_pattern = r'@\w+'

#balises html
html_pattern = r'<[^>]+>'

#url wite ou wwww 
url_pattern = r'\b(?:https?://|www\.)[^\s]+\b'

# espaces multiples
multiple_spaces_pattern = r'\s{2,}'

def clean_text(text):
    text = re.sub(email_pattern, '[EMAIL]', text)
    text = re.sub(phone_pattern, '[PHONE]', text)
    text = re.sub(date_pattern, '[DATE]', text)
    text = re.sub(prix_pattern, '[PRICE]', text)
    text = re.sub(hashtag_pattern, '', text)
    text = re.sub(mention_pattern, '', text)
    text = re.sub(html_pattern, '[HTML]', text)
    text = re.sub(url_pattern, '[URL]', text)
    text = re.sub(multiple_spaces_pattern, ' ', text)
    return text.strip()
for text in dataset:
    cleaned_text = clean_text(text)
    print(cleaned_text)

def extract_entities(text):
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    dates = re.findall(date_pattern, text)
    prices = re.findall(prix_pattern, text)
    hashtags = re.findall(hashtag_pattern, text)
    mentions = re.findall(mention_pattern, text)
    urls = re.findall(url_pattern, text)
    
    return {
        'emails': emails,
        'phones': phones,
        'dates': dates,
        'prices': prices,
        'hashtags': hashtags,
        'mentions': mentions,
        'urls': urls
    }
#"creat a dataframe to store the extracted entities"

df = pd.DataFrame(columns=['original_text', 'emails', 'phones', 'dates', 'prices', 'hashtags', 'mentions', 'urls'])
lignes = []
for text in dataset:
    entities = extract_entities(text)
    ligne = {
        'original_text': text,
        'emails': entities['emails'],
        'phones': entities['phones'],
        'dates': entities['dates'],
        'prices': entities['prices'],
        'hashtags': entities['hashtags'],
        'mentions': entities['mentions'],
        'urls': entities['urls']
    }
    lignes.append(ligne)
df = pd.DataFrame(lignes)
print(df.to_string())
#data= pd.DataFrame([extract_entities(text) for text in dataset])
# data.insert(0, 'original_text', dataset)
# print(data.to_string())