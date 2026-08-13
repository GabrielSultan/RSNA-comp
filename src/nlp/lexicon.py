# -*- coding: utf-8 -*-
"""Lexique multi-langue pour l'extraction des 12 cibles RSNA Knee.

Conventions :
- Le texte est normalise AVANT matching (voir extractor.normalize) :
  NFKC, minuscules, diacritiques retires pour toutes les langues SAUF le turc
  (ou i/I sont distinctives). Pour le turc : İ->i, I->ı puis lower().
- Les patterns turcs tolerent les variantes ASCII : [ıi], [üu], [öo], [şs], [çc], [ğg].
- Les patterns grecs sont ecrits SANS accent tonique (les diacritiques sont strippe).

Types de patterns par cible :
- "lesion"  : la mention du terme EST le finding (Effusion, Fracture, ...).
- "anatomy" : la cible est une structure ; le vote positif exige un mot-lesion
  proche dans la clause, le vote negatif un mot de normalite/negation proche.
- "oa"      : comme "anatomy" mais la lesion est un mot d'arthrose/chondropathie.
"""

LANGS = ["en", "es", "tr", "hr", "el", "de", "bg", "nl", "fr"]

# Langues a negation postfixee (le negateur suit le terme).
RIGHT_NEGATION = {"tr"}

# ---------------------------------------------------------------------------
# Negateurs verbaux (fenetre directionnelle : gauche, sauf RIGHT_NEGATION)
# ---------------------------------------------------------------------------
NEGATORS = {
    "en": [r"no", r"not", r"without", r"denies?", r"absen[ct]", r"free of",
           r"no evidence of", r"rules? out", r"negative for"],
    "es": [r"no", r"sin", r"ausencia", r"ausente", r"descarta[dn]?", r"niega",
           r"libre de", r"sin evidencia de"],
    "tr": [r"yok", r"yoktur", r"izlenme[md]", r"g[oö]r[üu]lme[md]",
           r"saptanma[md]", r"tespit edilme[md]", r"mevcut de[ğg]il",
           r"de[ğg]il", r"izlenmez"],
    "hr": [r"bez", r"nema", r"ne\b", r"nije", r"nisu", r"negira"],
    "el": [r"χωρισ", r"διχωσ", r"δεν", r"οχι", r"απουσια", r"απουσιαζει"],
    "de": [r"kein", r"keine[mnrs]?", r"ohne", r"nicht", r"fehlt",
           r"ausgeschlossen", r"ohne hinweis"],
    "bg": [r"без", r"няма", r"не се", r"отрича", r"липсва"],
    "nl": [r"geen", r"zonder", r"niet", r"afwezig", r"sluit uit"],
    "fr": [r"pas de", r"sans", r"absence", r"absente?", r"aucun", r"aucune",
           r"nie", r"pas d"],
}

# ---------------------------------------------------------------------------
# Mots de normalite (fenetre +-4 mots autour du terme) -> vote 0
# ---------------------------------------------------------------------------
NORMAL_WORDS = {
    "en": [r"normal", r"intact", r"preserved", r"unremarkable", r"stable",
           r"uninjured", r"unremarkale"],
    "es": [r"normal", r"normales", r"integr[ao]", r"conservad[ao]",
           r"indemne", r"sin alteraciones"],
    "tr": [r"normal", r"normaldir", r"korunmu[şs]", r"intakt", r"do[ğg]al",
           r"tabii", r"saglam"],
    "hr": [r"uredan", r"uredn[aeiou]", r"normalan", r"normaln[aeiou]",
           r"intaktan", r"intaktn[aeiou]", r"ocuvan", r"očuvan", r"redovit"],
    "el": [r"φυσιολογικ", r"ανεπαφ", r"ακεραι", r"διατηρ"],
    "de": [r"normal", r"intakt", r"unauffallig", r"unauffällig", r"erhalten",
           r"regelrecht", r"ohne befund"],
    "bg": [r"нормалн", r"непроменен", r"запазен", r"интактен", r"редовен"],
    "nl": [r"normaal", r"intact", r"ongeschonden", r"behouden"],
    "fr": [r"normal", r"normale", r"normaux", r"integre", r"conserve",
           r"indemne", r"respecte"],
}

# ---------------------------------------------------------------------------
# Mots d'incertitude (fenetre +-5 mots) -> confiance reduite
# ---------------------------------------------------------------------------
UNCERTAIN_WORDS = {
    "en": [r"possible", r"possibly", r"suspect", r"cannot exclude",
           r"may represent", r"probable", r"equivocal", r"question"],
    "es": [r"posible", r"posiblemente", r"sospecha", r"dudos[ao]",
           r"no se puede descartar", r"probable", r"inespecific"],
    "tr": [r"[şs][üu]phe", r"olas[ıi]", r"olabilir", r"muhtemel"],
    "hr": [r"moguce", r"moguće", r"sumnj", r"vjerojatno"],
    "el": [r"πιθαν", r"υποπτ", r"ενδεχομεν"],
    "de": [r"moglich", r"möglich", r"verdacht", r"fraglich", r"dd\b"],
    "bg": [r"възможн", r"съмнени", r"вероятн"],
    "nl": [r"mogelijk", r"verdenking", r"twijfel", r"verdacht"],
    "fr": [r"possible", r"suspect", r"doute", r"probable", r"eventuel"],
}

# ---------------------------------------------------------------------------
# Mots-lesion generiques (dechirure/rupture), partages ACL/MCL/menisques
# ---------------------------------------------------------------------------
TEAR_WORDS = {
    "en": [r"tears?", r"torn", r"ruptur\w*", r"sprain", r"discontinu\w+",
           r"transect\w*", r"macerat\w*", r"deficient", r"fissur\w*",
           r"lesion", r"injur\w*"],
    "es": [r"rotura", r"rot[oa]", r"ruptura", r"desgarr\w*", r"fisura",
           r"fissura", r"lesion", r"desgarro"],
    "tr": [r"y[ıi]rt[ıi]k", r"y[ıi]rt[ıi][ğg][ıi]", r"r[üu]pt[üu]r",
           r"kopuk", r"kopma", r"fiss[üu]r", r"lezyon"],
    "hr": [r"ruptur\w*", r"pukotin\w*", r"fisur\w*", r"kidanj\w*",
           r"nastrgan\w*", r"prekid", r"ozljed\w*"],
    "el": [r"ρηξη", r"ρηξ\w*", r"θλαση", r"ραγωδ\w*", r"βλαβη"],
    "de": [r"riss", r"ruptur\w*", r"fissur\w*", r"einriss", r"teilriss",
           r"lasion", r"läsion", r"verletzung"],
    "bg": [r"скъсване", r"разкъсване", r"надскъсване", r"фисура",
           r"увреждане", r"лезия"],
    "nl": [r"scheur\w*", r"ruptuur", r"fissuur", r"gescheurd", r"laesie",
           r"letsel"],
    "fr": [r"rupture", r"dechirure", r"fissure", r"lesion", r"dechire",
           r"entorse"],
}

# ---------------------------------------------------------------------------
# Mots-lesion d'arthrose / chondropathie (pour les 3 cibles OA)
# ---------------------------------------------------------------------------
OA_LESION = {
    "en": [r"osteoarthr\w*", r"arthros\w*", r"degenerat\w*",
           r"cartilage (loss|thinning|defect|wear|damage)",
           r"chondromalacia", r"chondros\w*", r"chondropath\w*",
           r"joint space narrow\w*", r"osteophyt\w*", r"denud\w+",
           r"chondral (loss|defect|damage|lesion)", r"osteoarthrit\w*"],
    "es": [r"artros\w*", r"artrotic\w*", r"degenera\w*", r"condromalacia",
           r"condropat\w*", r"pinzamiento", r"osteofit\w*", r"desnudacion",
           r"condral", r"perdida de cartilago"],
    "tr": [r"artroz", r"dejener\w*", r"k[ıi]k[ıi]rdak (kayb|kayb[ıi]|ince)",
           r"osteofit", r"kondromalazi", r"kondropati", r"kondral"],
    "hr": [r"artroz\w*", r"gonartroz\w*", r"degenera\w*", r"hondropatij\w*",
           r"denudacij\w*", r"osteofit\w*", r"hondromalacij\w*",
           r"hrskavic\w* (defekt|lezi)"],
    "el": [r"οστεοαρθρ\w*", r"αρθροσ\w*", r"εκφυλισ\w*", r"χονδρομαλακ\w*",
           r"οστεοφυτ\w*", r"χονδροπαθ\w*", r"γοναρθροσ\w*"],
    "de": [r"arthros\w*", r"gonarthros\w*", r"degener\w*", r"knorpel\w*",
           r"osteophyt\w*", r"chondromalazi\w*", r"chondropath\w*"],
    "bg": [r"артроз\w*", r"гонартроз\w*", r"дегенер\w*", r"хондромалация",
           r"остеофит\w*", r"хрущялн\w* (дефект|лезия)"],
    "nl": [r"artros\w*", r"degenerat\w*", r"kraakbeen\w*", r"osteofyten",
           r"chondromalaci\w*", r"chondropath\w*"],
    "fr": [r"arthros\w*", r"degener\w*", r"chondromalaci\w*", r"pincement",
           r"osteophyt\w*", r"chondropath\w*", r"gonarthros\w*"],
}

# ---------------------------------------------------------------------------
# Cibles "lesion pure" : la mention = le finding
# ---------------------------------------------------------------------------
LESION_TERMS = {
    "Effusion": {
        "en": [r"effusion", r"joint fluid", r"fluid (within|in) the (joint|knee)",
               r"h[ae]marthros\w*"],
        "es": [r"derrame", r"liquido articular", r"derramamiento",
               r"(lipo)?hemartros\w*"],
        "tr": [r"ef[üu]zyon", r"s[ıi]v[ıi] miktar[ıi]", r"s[ıi]v[ıi] art",
               r"mayi art", r"hemartroz"],
        "hr": [r"izliv", r"efuzij\w*", r"tekucin[ae] u zglobu",
               r"hemartroz\w*"],
        "el": [r"συλλογη υγρου", r"ενδαρθρικη συλλογη", r"υδραρθρ\w*",
               r"υγρο στην αρθρωση", r"αρθρικη συλλογη", r"αιμαρθρ\w*"],
        "de": [r"erguss", r"gelenkerguss", r"ergussbildung", r"hamarthros\w*",
               r"hämarthros\w*"],
        "bg": [r"излив", r"ставен излив", r"интраартикуларна течност"],
        "nl": [r"effusie", r"vocht (in|ter hoogte)", r"gewrichtsvocht"],
        "fr": [r"epanchement", r"epanch\w*"],
    },
    "Synovitis": {
        "en": [r"synovitis", r"synovial (proliferation|hypertrophy|thickening)",
               r"synovium hypertrophy", r"hypertroph\w+ synovium",
               r"thickened synovium"],
        "es": [r"sinovitis", r"hipertrofia sinovial"],
        "tr": [r"sinovit"],
        "hr": [r"sinovitis", r"sinovijalna proliferacija"],
        "el": [r"υμενιτιδα", r"συνοβιτιδα", r"αρθρομηνιτιδα"],
        "de": [r"synovitis", r"synovialitis"],
        "bg": [r"синовит"],
        "nl": [r"synovitis"],
        "fr": [r"synovite"],
    },
    "Baker's": {
        "en": [r"baker'?s cyst", r"popliteal cyst"],
        "es": [r"quiste\w* (de )?baker", r"quiste\w* popliteo"],
        "tr": [r"baker kist", r"popliteal kist"],
        "hr": [r"bakerov\w* cist", r"poplitealna cista"],
        "el": [r"κυστη baker", r"ποπλιτιαια κυστη", r"ποπλιτια κυστη"],
        "de": [r"baker.?zyste", r"poplitealzyste"],
        "bg": [r"бейкърова киста", r"поплитеална киста"],
        "nl": [r"baker.?cyste", r"popliteale cyste"],
        "fr": [r"kyste\w* (de )?baker", r"kyste\w* poplite"],
    },
    "Contusion": {
        "en": [r"bone (contusion|bruis\w+|marrow o?edema|o?edema)",
               r"marrow o?edema", r"bone o?edema", r"osseous contusion",
               r"bone bruise", r"subchondral (bone )?o?edema"],
        "es": [r"contusion osea", r"edema oseo", r"edema de (la )?medula osea",
               r"edema medular", r"contusion osea"],
        "tr": [r"kemik kont[üu]zyon", r"kemik ilig?i [oö]demi",
               r"kemik [oö]demi", r"kont[üu]zyon"],
        "hr": [r"kostan\w* kontuzij\w*", r"edem kosti", r"kontuzij\w* kosti",
               r"edem kostane srzi"],
        "el": [r"οστικο οιδημα", r"μωλωπη", r"οιδημα μυελου",
               r"μυελικο οιδημα"],
        "de": [r"knochenkontusion", r"knochenmarkodem", r"knochenmarködem",
               r"bone bruise", r"knochenodem"],
        "bg": [r"костна контузия", r"костномозъчен едем", r"костен едем"],
        "nl": [r"botcontusie", r"bot ?oedeem", r"mergoedeem",
               r"beenmergoedeem"],
        "fr": [r"contusion osseuse", r"oedeme osseux", r"meurtrissure",
               r"edeme osseux"],
    },
    "Fracture": {
        "en": [r"fracture", r"fractured", r"stress reaction"],
        "es": [r"fractura"],
        "tr": [r"k[ıi]r[ıi]k", r"frakt[üu]r"],
        "hr": [r"fraktur\w*", r"prijelom", r"lom\b"],
        "el": [r"καταγμα"],
        "de": [r"fraktur"],
        "bg": [r"фрактура"],
        "nl": [r"fractuur", r"breuk"],
        "fr": [r"fracture"],
    },
}

# ---------------------------------------------------------------------------
# Cibles anatomiques : vote 1 si mot-lesion proche, vote 0 si normalite/negation
# ---------------------------------------------------------------------------
ANATOMY_TERMS = {
    "ACL": {
        "en": [r"\bacl\b", r"anterior cruciate( ligament)?"],
        "es": [r"ligamentos? cruzados? anteriores?", r"\blca\b",
               r"cruzados?( y colaterales)?"],
        "tr": [r"[oö]n [çc]apraz ba[ğg]", r"\b[oö][çc]b\b",
               r"[çc]apraz ba[ğg]"],
        "hr": [r"prednji\w* krizn\w* ligament", r"križn\w* ligament",
               r"krizn\w* ligament", r"\bacl\b"],
        "el": [r"προσθιο\w* χιαστ\w*", r"χιαστοσ"],
        "de": [r"vordere\w* kreuzband", r"\bvkb\b", r"\bacl\b"],
        "bg": [r"предна кръстна връзка"],
        "nl": [r"voorste kruisband", r"\bvkb\b", r"\bacl\b"],
        "fr": [r"ligament croise anterieur", r"\blca\b", r"croise anterieur"],
    },
    "MCL": {
        "en": [r"\bmcl\b", r"medial collateral( ligament)?"],
        "es": [r"ligamento colateral (medial|interno)",
               r"colateral (medial|interno)"],
        "tr": [r"medial kollateral", r"i[çc] yan ba[ğg]",
               r"medial yan ba[ğg]", r"\bmcl\b"],
        "hr": [r"medijalni\w* kolateraln\w*", r"unutarnji\w* bocn\w*",
               r"\bmcl\b"],
        "el": [r"εσω πλαγιο\w* συνδεσμ\w*", r"εσω πλαγιο\w*"],
        "de": [r"\bmkl\b", r"mediale\w* kollateral", r"innere\w* seitenband"],
        "bg": [r"медиална колатерална връзка", r"вътрешна странична връзка"],
        "nl": [r"\bmcl\b", r"mediale collaterale", r"binnenste zijband"],
        "fr": [r"ligament collateral medial", r"\blcm\b",
               r"collateral interne"],
    },
    "Medial Meniscus": {
        "en": [r"medial( and lateral)? menisc\w*", r"medial menisc\w*"],
        "es": [r"meniscos? (medial|interno|mediales)",
               r"menisco medial y lateral"],
        "tr": [r"medial menisk\w*", r"medyal menisk\w*"],
        "hr": [r"medijaln\w* menisk\w*", r"unutarnji\w* menisk\w*"],
        "el": [r"εσω μηνισκ\w*"],
        "de": [r"innenmenisk\w*", r"medial\w* menisk\w*", r"medialmenisk\w*"],
        "bg": [r"медиалн\w* менискус\w*"],
        "nl": [r"mediale menisc\w*", r"binnenmenisc\w*"],
        "fr": [r"menisque (medial|interne)", r"menisque medial"],
    },
    "Lateral Meniscus": {
        "en": [r"lateral menisc\w*", r"lateral and medial menisc\w*"],
        "es": [r"meniscos? (lateral|externo)"],
        "tr": [r"lateral menisk\w*"],
        "hr": [r"lateraln\w* menisk\w*", r"vanjski\w* menisk\w*"],
        "el": [r"εξω μηνισκ\w*"],
        "de": [r"aussenmenisk\w*", r"lateral\w* menisk\w*",
               r"lateralmenisk\w*"],
        "bg": [r"латералн\w* менискус\w*"],
        "nl": [r"laterale menisc\w*", r"buitenmenisc\w*"],
        "fr": [r"menisque (lateral|externe)"],
    },
}

# ---------------------------------------------------------------------------
# Cibles OA : anatomy = compartiment, lesion = OA_LESION
# ---------------------------------------------------------------------------
OA_ANATOMY = {
    "Medial OA": {
        "en": [r"medial (tibiofemoral )?(compartment|joint space|femoral condyle|tibial plateau)",
               r"tricompartment\w*"],
        "es": [r"(compartimento|compartimiento|articulacion) (femorotibial )?medial",
               r"femorotibial medial", r"condilo femoral medial",
               r"tricompartimental"],
        "tr": [r"medial (kompartman|eklem|tibiofemoral|femoral kondil|tibial plato)",
               r"medyal (kompartman|eklem)"],
        "hr": [r"medijaln\w* (kompartment|femorotibijaln\w*|kondil\w*|tibijaln\w*)"],
        "el": [r"εσω (μηροκνημιδιο\w*|διαμερισμα\w*|κοτυλ\w*)",
               r"εσω κνημ\w*"],
        "de": [r"mediale\w* (kompartiment|femorotibial\w*|kondylus|tibiaplateau)",
               r"medialkompartiment"],
        "bg": [r"медиал\w* (компартмент|феморотибиал\w*|кондил|плато)"],
        "nl": [r"mediale (compartiment|femorotibiale|femurcondyl|tibiaplateau)"],
        "fr": [r"(compartiment|articulation) (femorotibial )?medial",
               r"femorotibial medial", r"condyle femoral medial"],
    },
    "Lateral OA": {
        "en": [r"lateral (tibiofemoral )?(compartment|joint space|femoral condyle|tibial plateau)",
               r"tricompartment\w*"],
        "es": [r"(compartimento|compartimiento|articulacion) (femorotibial )?lateral",
               r"femorotibial lateral", r"condilo femoral lateral",
               r"tricompartimental"],
        "tr": [r"lateral (kompartman|eklem|tibiofemoral|femoral kondil|tibial plato)"],
        "hr": [r"lateraln\w* (kompartment|femorotibijaln\w*|kondil\w*|tibijaln\w*)"],
        "el": [r"εξω (μηροκνημιδιο\w*|διαμερισμα\w*)", r"εξω κνημ\w*"],
        "de": [r"laterale\w* (kompartiment|femorotibial\w*|kondylus|tibiaplateau)",
               r"lateralkompartiment"],
        "bg": [r"латерал\w* (компартмент|феморотибиал\w*|кондил|плато)"],
        "nl": [r"laterale (compartiment|femorotibiale|femurcondyl|tibiaplateau)"],
        "fr": [r"(compartiment|articulation) (femorotibial )?lateral",
               r"femorotibial lateral", r"condyle femoral lateral"],
    },
    "PF OA": {
        "en": [r"patellofemoral", r"patellar cartilage", r"retropatellar",
               r"trochlea", r"tricompartment\w*"],
        "es": [r"patelofemoral", r"rotulian\w*", r"femoropatelar",
               r"condromalacia rotuliana", r"tricompartimental"],
        "tr": [r"patellofemoral", r"patella"],
        "hr": [r"patelofemoraln\w*", r"patel\w*"],
        "el": [r"επιγονατιδομηριαιο\w*", r"επιγονατιδ\w*"],
        "de": [r"patellofemoral", r"retropatellar", r"femoropatellar"],
        "bg": [r"пателофеморалн\w*", r"пател\w*"],
        "nl": [r"patellofemoraal", r"retropatellair", r"femoropatellair"],
        "fr": [r"patellofemoral", r"femoropatellaire", r"rotulien"],
    },
}

# ---------------------------------------------------------------------------
# Sections a NE PAS lire : antecedents / indication / technique.
# Les mentions y sont du contexte clinique, pas des findings.
# Regex matchees en debut de ligne (texte normalise, minuscules).
# ---------------------------------------------------------------------------
SECTION_SKIP = {
    "en": [r"indication", r"clinical (history|indication)", r"history\b",
           r"comparison", r"technique", r"exam(type|ination)?\b", r"procedure"],
    "es": [r"antecedentes", r"indicacion", r"historia clinica", r"motivo",
           r"tecnica"],
    "tr": [r"klinik", r"anamnez", r"endikasyon", r"tetkik", r"hikaye"],
    "hr": [r"anamneza", r"indikacija", r"tehnika", r"usporedba"],
    "el": [r"αναμνησ", r"ενδειξ", r"τεχνικη", r"κλινικ"],
    "de": [r"anamnese", r"indikation", r"klinisch", r"technik",
           r"vorbefund", r"vergleich"],
    "bg": [r"анамнеза", r"индикаци", r"клиничн", r"техника"],
    "nl": [r"indicatie", r"anamnese", r"klachten", r"techniek",
           r"vergelijking"],
    "fr": [r"indication", r"antecedents", r"histoire", r"motif", r"technique"],
}

# En-tetes qui TERMINENT le saut (reprise de la lecture)
SECTION_RESUME = {
    "en": [r"findings", r"impression", r"conclusion", r"report"],
    "es": [r"hallazgos", r"impresion", r"conclusion", r"resultados"],
    "tr": [r"bulgular", r"sonuc", r"impression"],
    "hr": [r"nalaz", r"zakljucak", r"opis"],
    "el": [r"ευρηματα", r"συμπερασμα"],
    "de": [r"beurteilung", r"befund", r"zusammenfassung", r"fazit"],
    "bg": [r"находка", r"заключение", r"извод"],
    "nl": [r"bevindingen", r"conclusie", r"impressie"],
    "fr": [r"resultats", r"conclusion", r"impression"],
}

# Ordre canonique des cibles (colonnes de train.csv / submission)
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

ANATOMY_TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"]
OA_TARGETS = ["Medial OA", "Lateral OA", "PF OA"]
LESION_TARGETS = ["Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
