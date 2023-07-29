

* idade materna 
*(1= ≤17; 2= 18 e 19; 0= 20 -34; 3= ≥35
gen idadem1 =1
replace idadem1 =2 if (iadedem >=18 & iadedem<20)
replace idadem1 =0 if (iadedem >=20 & iadedem<35)
replace idadem1 =3 if iadedem >=35


label define idadem1 1 "≤17" 2 "18-19" 0 "20-34" 3 "≥35"
label values idadem1 idadem1



* Estado civil da mae 
* 0=casada, união estável e  1=solteira, viúva ou separada

gen estadocivil =0
replace estadocivil =1 if estcivmae ==1
replace estadocivil =1 if estcivmae ==3
replace estadocivil =1 if estcivmae ==4
replace estadocivil =9 if estcivmae ==9
replace estadocivil=. if estcivmae==9
label define estadocivil 0 "casada, união estável" 1 "solteira, viúva ou separada"
label values estadocivil estadocivil
 

*Escolaridade da mae

gen escola_mae=escmae
recode escola_mae 1=1 2=2 3=3 4=4 5=0
replace escola_mae=. if escmae==9
label define escola_mae 1"Nenhuma" 2 "1 a 3 anos" 3"4 a 7 anos"  4"8 a 11 anos" 0"12 e mais"
label values escola_mae escola_mae

* Semana gestacional da gestante 

gen semanagest = gestacao
recode semanagest 2=2 3=3 4=4 5=5 6=0
label define  semanagest 2"22 a 27 semanas" 3"28 a 31 semanas" 4"32 a 36 semanas" 0"37 a 42 semanas"
label values semanagest semanagest

* Numero de Consultas prenatais 

gen Consulta_pre = consultas
replace Consulta_pre=. if consultas==9
recode Consulta_pre 4=0 1=1 2=2 3=3 
label define Consulta_pre 1"Nenhuma"  2 "de 1 a 3" 3"de 4 a 6" 4"7 e mais"
label values Consulta_pre Consulta_pre


*sexo do recem nascido
*1=masculino; 2=feminino)
gen sexo_recem = sexo
label define sexo_recem 1"Masculino" 2"Femenino"
label values sexo_recem sexo_recem 

* Raca/cor da mae 

gen raca_mae = racacor

label define raca_mae 1"Branca" 2"Preta" 3"Amarela" 4"Parda" 5"Indigena"
label values raca_mae raca_mae
replace raca_mae =. if racacor==9

*ROturar a densidade de alimentos nao saudaveis 

label define Ali_Nsaud1 1"1º tercil (<p33,3)" 2 "2º tercil (>=p33,3; =<p66,6)" 3"3º tercil (>p66,6)"
label values Ali_Nsaud1 Ali_Nsaud1

*ROturar a densidade de alimentos nao saudaveis 

label define Ali_Saud 3"1º tercil (<p33,3)" 2 "2º tercil (>=p33,3; =<p66,6)" 1"3º tercil (>p66,6)"
label values Ali_Saud Ali_Saud

*Regressao logistica com o modelo 
logistic bpn i.Ali_Nsaud1 i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae
mlogit bwga i.Ali_Nsaud1 i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae,rr
logistic gig i.Ali_Nsaud1 i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae idhm pibpcapita renda_gini desemprego


logistic gig i.Ali_Nsaud1 i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae idhm pibpcapita renda_gini desemprego
logistic pig i.Ali_Nsaud1 i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae idhm pibpcapita renda_gini desemprego
logistic bpn i.Ali_Nsaud1 i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae idhm pibpcapita renda_gini desemprego
logistic pig i.Ali_saude i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae idhm pibpcapita renda_gini desemprego 
logistic bpn i.Ali_saude i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae idhm pibpcapita renda_gini desemprego 
logistic gig i.Ali_saude i.idadem1 i.estadocivil i.escola_mae i.Consulta_pre i.sexo_recem i.raca_mae idhm pibpcapita renda_gini desemprego 

gen regiao=Regiao
label define  regiao 1"norte" 2"nordeste" 3"sudeste"  4"sul" 5"centro oeste"
label values regiao1 regiao

,
