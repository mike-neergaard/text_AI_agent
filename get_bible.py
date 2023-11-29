from meaningless import WebExtractor

def get_book(bible, book):
   print("retrieving",book,"...", flush=True, end="")
   passage = bible.get_book(book)
   if len(passage) > 0: 
       print("success!\n  Writing...", flush=True, end="")
       with open(book+".txt", "w") as outfile: 
           outfile.write(passage)
       print("done!")
   else:
       print("Failure!")


old_testament = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Songs", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"]
old_testament = ["Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"]

new_testament = ["Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"]

bible=WebExtractor(translation='NRSVUE')

for book in old_testament:
    get_book(bible, book)

for book in new_testament:
    get_book(bible, book)
