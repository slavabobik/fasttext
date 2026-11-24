package langid_test

// samples holds one or two sentences for each of 79 languages in 27 writing
// systems. Most say the same thing, a table booking for two this evening, so
// that the languages differ and the content does not. TestDetect expects
// both lid.176 models to name each language, except for closeNeighbours.
// BenchmarkDetectScript times them per writing system.
var samples = []struct{ script, code, text string }{
	// Latin
	{"Latin", "en", "Where is my order? It was supposed to arrive yesterday."},
	{"Latin", "en", "Good evening, I would like to book a table for two people tonight."},
	{"Latin", "es", "¿Dónde está mi pedido? Tenía que llegar ayer."},
	{"Latin", "es", "Buenas tardes, quisiera reservar una mesa para dos personas esta noche."},
	{"Latin", "pt", "Onde está o meu pedido? Era para chegar ontem."},
	{"Latin", "pt", "Boa noite, gostaria de reservar uma mesa para duas pessoas hoje à noite."},
	{"Latin", "fr", "Bonjour tout le monde, comment allez-vous ?"},
	{"Latin", "fr", "Bonsoir, je voudrais réserver une table pour deux personnes ce soir."},
	{"Latin", "it", "Dov'è il mio ordine? Doveva arrivare ieri."},
	{"Latin", "it", "Buonasera, vorrei prenotare un tavolo per due persone per stasera."},
	{"Latin", "ca", "Bon vespre, voldria reservar una taula per a dues persones aquesta nit."},
	{"Latin", "gl", "Onte fomos ao mercado e mercamos moitas cousas para a casa nova."},
	{"Latin", "ro", "Bună seara, aș dori să rezerv o masă pentru două persoane în seara asta."},
	{"Latin", "la", "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae."},
	{"Latin", "de", "Wo ist meine Bestellung? Sie sollte gestern ankommen."},
	{"Latin", "de", "Guten Abend, ich möchte gern einen Tisch für zwei Personen reservieren."},
	{"Latin", "nl", "Mijn bestelling is nog steeds niet aangekomen."},
	{"Latin", "nl", "Goedenavond, ik wil graag een tafel voor twee personen reserveren voor vanavond."},
	{"Latin", "af", "Goeie naand, ek wil graag vanaand 'n tafel vir twee mense bespreek."},
	{"Latin", "sv", "God kväll, jag skulle vilja boka ett bord för två personer i kväll."},
	{"Latin", "da", "God aften, jeg vil gerne bestille et bord til to personer i aften."},
	{"Latin", "no", "God kveld, jeg vil gjerne bestille et bord til to personer i kveld."},
	{"Latin", "is", "Gott kvöld, mig langar að panta borð fyrir tvo í kvöld."},
	{"Latin", "pl", "Gdzie jest moje zamówienie? Miało przyjść wczoraj."},
	{"Latin", "pl", "Dobry wieczór, chciałbym zarezerwować stolik dla dwóch osób na dzisiejszy wieczór."},
	{"Latin", "cs", "Dobrý večer, chtěl bych si rezervovat stůl pro dvě osoby na dnešní večer."},
	{"Latin", "sk", "Dobrý večer, chcel by som si rezervovať stôl pre dve osoby na dnešný večer."},
	{"Latin", "sl", "Dober večer, rad bi rezerviral mizo za dve osebi za nocoj."},
	{"Latin", "hr", "Dobra večer, želio bih rezervirati stol za dvije osobe za večeras."},
	{"Latin", "lt", "Labas vakaras, norėčiau užsisakyti staliuką dviem asmenims šį vakarą."},
	{"Latin", "lv", "Labvakar, es vēlētos rezervēt galdiņu divām personām šovakar."},
	{"Latin", "hu", "Jó estét, szeretnék asztalt foglalni két személyre ma estére."},
	{"Latin", "fi", "Hyvää iltaa, haluaisin varata pöydän kahdelle hengelle tälle illalle."},
	{"Latin", "et", "Tere õhtust, ma sooviksin broneerida laua kahele inimesele tänaseks õhtuks."},
	{"Latin", "eu", "Arratsalde on, gaur gauerako bi pertsonentzako mahai bat erreserbatu nahi nuke."},
	{"Latin", "ga", "Tráthnóna maith, ba mhaith liom bord a chur in áirithe do bheirt anocht."},
	{"Latin", "cy", "Noswaith dda, hoffwn i archebu bwrdd i ddau o bobl heno os gwelwch yn dda."},
	{"Latin", "sq", "Mirëmbrëma, do të doja të rezervoja një tavolinë për dy persona për sonte."},
	{"Latin", "mt", "Bonswa, nixtieq nirriżerva mejda għal żewġ persuni għal-lejla."},
	{"Latin", "tr", "Siparişim nerede? Dün gelmesi gerekiyordu."},
	{"Latin", "tr", "İyi akşamlar, bu akşam için iki kişilik bir masa ayırtmak istiyorum."},
	{"Latin", "az", "Axşamınız xeyir, bu axşam üçün iki nəfərlik masa sifariş etmək istəyirəm."},
	{"Latin", "uz", "Xayrli kech, bugun kechqurun uchun ikki kishilik stol band qilmoqchiman."},
	{"Latin", "vi", "Chào buổi tối, tôi muốn đặt một bàn cho hai người vào tối nay."},
	{"Latin", "id", "Selamat malam, saya ingin memesan meja untuk dua orang untuk nanti malam."},
	{"Latin", "ms", "Selamat petang, saya hendak menempah sebuah meja untuk dua orang pada malam ini."},
	{"Latin", "tl", "Magandang gabi po, gusto ko sanang magpareserba ng mesa para sa dalawang tao mamayang gabi."},
	{"Latin", "sw", "Habari za jioni, ningependa kuhifadhi meza kwa ajili ya watu wawili usiku wa leo."},
	{"Latin", "eo", "Bonan vesperon, mi ŝatus rezervi tablon por du personoj por ĉi tiu vespero."},

	// Cyrillic
	{"Cyrillic", "ru", "Где мой заказ? Он должен был прийти вчера."},
	{"Cyrillic", "ru", "Добрый вечер, я хотел бы забронировать столик на двоих на сегодня."},
	{"Cyrillic", "uk", "Де моє замовлення? Воно мало прийти вчора."},
	{"Cyrillic", "uk", "Добрий вечір, я хотів би забронювати столик на двох на сьогодні."},
	{"Cyrillic", "be", "Добры вечар, я хацеў бы забраніраваць столік на дваіх на сёння."},
	{"Cyrillic", "bg", "Добър вечер, бих искал да запазя маса за двама за тази вечер."},
	{"Cyrillic", "mk", "Добра вечер, би сакал да резервирам маса за двајца за вечерва."},
	{"Cyrillic", "sr", "Добро вече, желео бих да резервишем сто за двоје за вечерас."},
	{"Cyrillic", "kk", "Қайырлы кеш, бүгін кешке екі адамға арналған үстел брондағым келеді."},
	{"Cyrillic", "mn", "Оройн мэнд, өнөө орой хоёр хүний ширээ захиалмаар байна."},

	// Other alphabets of Europe and the Caucasus
	{"Greek", "el", "Καλησπέρα, θα ήθελα να κλείσω ένα τραπέζι για δύο άτομα για απόψε."},
	{"Armenian", "hy", "Բարի երեկո, ես կցանկանայի այսօր երեկոյան երկու հոգու համար սեղան պատվիրել։"},
	{"Georgian", "ka", "საღამო მშვიდობისა, მინდა დავჯავშნო მაგიდა ორი ადამიანისთვის დღეს საღამოს."},

	// Hebrew and Arabic scripts
	{"Hebrew", "he", "ערב טוב, אני רוצה להזמין שולחן לשני אנשים להערב."},
	{"Hebrew", "yi", "גוטן אָוונט, איך װיל באַשטעלן אַ טיש פֿאַר צוויי מענטשן פֿאַר הײַנט בײַ נאַכט."},
	{"Arabic", "ar", "أين طلبي؟ كان من المفترض أن يصل أمس."},
	{"Arabic", "ar", "مساء الخير، أود أن أحجز طاولة لشخصين لهذه الليلة."},
	{"Arabic", "fa", "عصر بخیر، می\u200cخواهم برای امشب یک میز برای دو نفر رزرو کنم."},
	{"Arabic", "ur", "شام بخیر، میں آج رات کے لیے دو لوگوں کی ایک میز بک کرنا چاہتا ہوں۔"},
	{"Arabic", "ug", "كەچلىك سالام، مەن بۈگۈن كەچتە ئىككى كىشىلىك بىر ئۈستەل زاكاز قىلماقچىمەن."},

	// Scripts of South Asia
	{"Devanagari", "hi", "मेरा ऑर्डर कहाँ है? इसे कल आना था।"},
	{"Devanagari", "hi", "शुभ संध्या, मैं आज रात के लिए दो लोगों की एक मेज़ बुक करना चाहता हूँ।"},
	{"Devanagari", "mr", "शुभ संध्याकाळ, मला आज रात्रीसाठी दोन लोकांसाठी एक टेबल आरक्षित करायचे आहे."},
	{"Devanagari", "ne", "शुभ साँझ, म आज बेलुकाको लागि दुई जनाको टेबल बुक गर्न चाहन्छु।"},
	{"Bengali", "bn", "শুভ সন্ধ্যা, আমি আজ রাতের জন্য দুজনের একটি টেবিল সংরক্ষণ করতে চাই।"},
	{"Gurmukhi", "pa", "ਸ਼ੁਭ ਸ਼ਾਮ, ਮੈਂ ਅੱਜ ਰਾਤ ਲਈ ਦੋ ਲੋਕਾਂ ਵਾਸਤੇ ਇੱਕ ਮੇਜ਼ ਬੁੱਕ ਕਰਨਾ ਚਾਹੁੰਦਾ ਹਾਂ।"},
	{"Gujarati", "gu", "શુભ સાંજ, હું આજે રાત માટે બે લોકો માટે એક ટેબલ બુક કરવા માંગુ છું."},
	{"Odia", "or", "ଶୁଭ ସନ୍ଧ୍ୟା, ମୁଁ ଆଜି ରାତି ପାଇଁ ଦୁଇ ଜଣଙ୍କ ପାଇଁ ଗୋଟିଏ ଟେବୁଲ ବୁକ୍ କରିବାକୁ ଚାହୁଁଛି।"},
	{"Tamil", "ta", "மாலை வணக்கம், இன்று இரவுக்கு இரண்டு பேருக்கு ஒரு மேசையை முன்பதிவு செய்ய விரும்புகிறேன்."},
	{"Telugu", "te", "శుభ సాయంత్రం, ఈ రోజు రాత్రికి ఇద్దరి కోసం ఒక టేబుల్ బుక్ చేయాలనుకుంటున్నాను."},
	{"Kannada", "kn", "ಶುಭ ಸಂಜೆ, ನಾನು ಇಂದು ರಾತ್ರಿಗೆ ಇಬ್ಬರಿಗಾಗಿ ಒಂದು ಟೇಬಲ್ ಕಾಯ್ದಿರಿಸಲು ಬಯಸುತ್ತೇನೆ."},
	{"Malayalam", "ml", "ശുഭ സായാഹ്നം, ഇന്ന് രാത്രിയിലേക്ക് രണ്ട് പേർക്ക് ഒരു മേശ ബുക്ക് ചെയ്യാൻ ഞാൻ ആഗ്രഹിക്കുന്നു."},
	{"Sinhala", "si", "සුබ සන්ධ්\u200dයාවක්, මට අද රාත්\u200dරියට දෙදෙනෙකු සඳහා මේසයක් වෙන් කර ගැනීමට අවශ්\u200dයයි."},
	{"Tibetan", "bo", "དགོང་དྲོ་བདེ་ལེགས། ང་དོ་དགོང་མི་གཉིས་ཀྱི་ཆེད་དུ་ཅོག་ཙེ་གཅིག་སྔ་གཏན་འཇོག་བྱེད་འདོད་ཡོད།"},
	{"Thaana", "dv", "ހަވީރު ބަޚައިރު، މިރޭ ދެމީހުންނަށް މޭޒެއް ބުކް ކުރަން ބޭނުން."},

	// Scripts of Southeast Asia
	{"Thai", "th", "สวัสดีตอนเย็นครับ ผมอยากจองโต๊ะสำหรับสองคนคืนนี้"},
	{"Lao", "lo", "ສະບາຍດີຕອນແລງ, ຂ້ອຍຢາກຈອງໂຕະສຳລັບສອງຄົນໃນຄືນນີ້."},
	{"Khmer", "km", "សាយ័ណ្ហសួស្តី ខ្ញុំចង់កក់តុមួយសម្រាប់មនុស្សពីរនាក់នៅយប់នេះ។"},
	{"Myanmar", "my", "မင်္ဂလာညနေခင်းပါ၊ ဒီည လူနှစ်ယောက်အတွက် စားပွဲတစ်လုံး ကြိုတင်မှာယူချင်ပါတယ်။"},

	// East Asia
	{"Han", "zh", "我的订单还没有到。"},
	{"Han", "zh", "晚上好，我想预订今晚两个人的桌子。"},
	{"Han", "zh", "晚安，我想預訂今晚兩個人的位子。"},
	{"Japanese", "ja", "注文した商品がまだ届いていません。"},
	{"Japanese", "ja", "こんばんは、今夜二人で席を予約したいのですが。"},
	{"Hangul", "ko", "주문한 상품이 아직 도착하지 않았습니다."},
	{"Hangul", "ko", "안녕하세요, 오늘 저녁에 두 사람 자리를 예약하고 싶습니다."},

	// Africa
	{"Ethiopic", "am", "እንደምን አመሹ፣ ዛሬ ማታ ለሁለት ሰዎች ጠረጴዛ ማስያዝ እፈልጋለሁ።"},
}
