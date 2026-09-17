dataset = [
    "Bonjour, mon email est   omar.k@example.com   et mon tel est 0612345678.",
    "Contactez-nous via www.monsite.ma ou https://monsite.ma/contact !!",
    "Le prix est 250 MAD le 17/09/2026, livraison rapide #NLP #IA @client_service",
    "<p>Merci pour votre commande</p>   <br> Numero: +212 6-61-22-33-44",
    "  Service   nul...    remboursement demande le 05-08-2026  , prix 1200,50 MAD ",
    "Great service! Contact: sales@company.com or call 0522-334455 #satisfied",
    "Facture N° INV-2026-00457 total: 3999.99 MAD TVA 20% date: 2026/09/01",
    "Aucun probleme, tout est parfait :) suivez-nous @companyOfficial https://x.com/companyOfficial"
]

# email : partie_local @ domaine . extension
import re
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'

#tele : indicateur c'est +212 ou 0 suivi de 9 chiffres
phone_pattern = r'\b(?:\+212|0)\d{9}\b'

#date JJ/MM/AAAA ou AAAA/MM/JJ ou JJ-MM-AAAA ou AAAA-MM-JJ
date_pattern = r'\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b'

#prix

#hachtags et mentions

#balises html

#url

#espaces