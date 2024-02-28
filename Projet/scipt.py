text = open("./pos_reference.txt").read()
text = text.split("\n")
newText = []
tagsRef = []
for t in text:
    t = t.split("\t")
    newText.append(t[0])
    if(len(t)>1):
        tagsRef.append(t[1])

string = ""


for t in newText:
    if(t!=''):
        string = string + " " + t
    else:
        string = string + "\n"

with open("pos_text.txt",'w') as f:
    f.write(string)

##On cree une grille qui passe de tag REF à des tags PTB
tagRefPtb = open('./POSTags_REF_PTB.txt').read()
tagRefPtb = tagRefPtb.split("\n")
for i in range(len(tagRefPtb)):
    tagRefPtb[i] = tagRefPtb[i].split()


##On cree une grille qui passe de tag PTB à des tags Universal
tagPtbUni = open('./POSTags_PTB_Universal.txt').read()
tagPtbUni = tagPtbUni.split("\n")
for i in range(len(tagPtbUni)):
    tagPtbUni[i] = tagPtbUni[i].split()

tagsUni = ""
for tagRef in tagsRef:
    for i in range(len(tagRefPtb)):
        if(tagRefPtb[i][0] == tagRef):
            tagPTB = tagRefPtb[i][1]
    
    for i in range(len(tagPtbUni)):
        if(tagPtbUni[i][0] == tagPTB):
            tagUni = tagPtbUni[i][1]

    tagsUni = tagsUni + tagUni + "\n"

with open("pos_reference.txt.univ",'w') as f:
    f.write(tagsUni)


