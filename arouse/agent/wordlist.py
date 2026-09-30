"""Word pools for "wild" training text: unpredictable task/note phrases that can only be
reproduced by copying the user's words (not by recalling memorised phrases).

Deliberately excludes time/day/scheduling words so random phrases never change the
meaning of a request.
"""

VERBS = """call email text visit meet pay buy order check clean fix book send collect renew review update prepare
pick drop print sign submit cancel return wash repair inspect count feed water charge read finish start plan
confirm measure weigh label pack unpack move paint polish sharpen defrost test replace install remove
record photograph scan copy file sort stack load unload deliver fetch borrow lend sell trade register apply
practice revise study write draft translate calculate compare estimate verify approve reject archive
mend sew iron cook bake boil fry chop peel grind mix stir brew pour serve tidy sweep mop dust vacuum empty fill
refill lock unlock open close tighten loosen oil grease adjust align calibrate tune plant prune harvest spray
dig rake mow trim cut split stitch glue tape wrap ship post mail invite thank congratulate greet visit""".split()

NOUNS = """invoice receipt ledger budget report contract letter parcel package bottle can bucket barrel sack bag
box crate pallet shelf rack cabinet drawer folder binder notebook pen pencil marker printer scanner laptop
monitor keyboard mouse cable charger battery bulb lamp switch socket plug fuse wire pipe hose valve tap
pump motor engine tractor trailer truck van bike scooter tyre wheel brake chain belt gear lever handle hinge
gate fence wall roof door window floor ceiling stairs ladder shed barn stable coop kennel garden lawn hedge
tree bush flower seed bulb pot soil compost manure fodder hay straw grain maize wheat rice lentil bean pea
onion potato tomato chilli garlic ginger lemon mango banana apple milk curd butter ghee cheese paneer cream
yogurt egg bread flour sugar salt oil tea coffee spice cow calf buffalo goat sheep hen dog cat horse bull
heifer herd vet doctor nurse teacher driver farmer trader broker banker lawyer accountant clerk cashier
tailor plumber electrician carpenter mechanic painter cleaner guard neighbour cousin uncle aunt nephew niece
friend partner customer supplier dealer agent landlord tenant school college library clinic hospital
pharmacy bank market shop office warehouse depot station airport temple stadium museum cinema bakery
salon garage workshop factory dairy farm field pond well tank canal river bridge road lane corner
medicine tablet syrup vaccine bandage glove mask helmet boot jacket shirt trouser saree scarf blanket
pillow mattress curtain carpet towel soap shampoo brush comb mirror clock watch phone radio camera album
photo ticket passport licence permit certificate form application loan cheque pension subsidy insurance
premium bill fee fine tax rent salary bonus wage refund deposit payment transfer balance statement
password account email message invitation gift card cake candle ribbon balloon flag banner poster sign
map guide manual recipe menu list timetable agenda notes summary draft proposal quote
sample specimen result reading meter gauge scale thermometer sensor alarm camera filter nozzle sprayer""".split()

ADJECTIVES = """new old spare broken clean dirty empty full large small heavy light red blue green yellow black
white brown silver golden wooden plastic metal glass paper cotton wool leather steel copper extra overdue
pending urgent final rough fresh frozen dried organic local imported main back front side upper lower north
south east west inner outer long short wide narrow""".split()

NAMES = """Aarav Aditi Akash Alok Ananya Anil Anjali Arjun Asha Bhavna Chetan Deepa Dev Divya Esha Farah Gaurav
Geeta Hari Hema Imran Indu Ishaan Jatin Jyoti Kabir Kamala Karan Lata Madhav Maya Mohan Nandini Naveen Neel
Nikita Omkar Pallavi Pankaj Parvati Pradeep Rachna Raghav Rekha Rohit Sakshi Sameer Sanjay Seema Shankar
Shreya Siddharth Sneha Tara Tarun Uday Uma Varun Veena Vijay Yash Zara John Maria Ahmed Chen Fatima Lucas""".split()

SYLLABLES = ["ka", "ren", "to", "vi", "lo", "mar", "sa", "den", "ru", "pel", "ti", "zo", "bar", "nu", "fel", "qui",
             "dra", "mo", "kel", "shi", "pan", "gor", "li", "tas", "ve", "ron", "da", "mi", "sul", "ek"]
