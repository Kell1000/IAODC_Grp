dataset = [
    " Bonjour mon email est omar.kexample@gmail.com et mon tel est 0612345678",
    "contactez-nous via www.monsite.ma ou https//monsite.ma/contact!!",
    "le prix est 250 MAD le 17/09/2026, livraison rapide #IA@client_service",
    "<p>Merci pour votre commande</p> <br>Numero:+2126-61-22-33-44,"
    "Service nul... remboursement demande le 05-08-2026 ,prix 1200,50 MAD,"
    "Great service!contact:sale@company.com or call 0522-334455#satifietd,"
    "Facture N° INV-2026-00457 total:3999.99 MAD, TVA 20% date:2026/09/01,"
    "Aucun probleme, tout est parfait :)suivez-vous @companyOfficialhttps://x.com/companyOfficial"

]

# email:patrtie_local @ domaine . expension
import re
email_pattern = r'/b[A-Za-z0-9._%+-]+@[a-z0-9,-]+/,[a-zA-Z]{2,}/b'
#date : JJ/MM/AAAA ou AAAA/MM/JJ ou JJ-MM-AAAA OU AAAA-MM-JJ
date_pattern = r'/b(?:/d{2}[/-]/d{2}[/-]/d{4}|/d{4}[/-]/d{2}[/-]/d{2})/d{2}-/d{2}-/d{4}/{4}-/d{2}-/d{2}/d'
#prix
prix_pattern = 
