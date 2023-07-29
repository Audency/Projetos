
*Regressao de Poisson
poisson 

regressao binomial negativa 
nbreg txmort_dcv Ali_saud11 Ali_saud22 Ali_Nsaud1, irr

Densid_innatura1 Densid_Mistros1 Densid_ultraprocessados1

* Calculo da densidade(2010)
gen Densid_innatura=( NumdeestabelnNatura/pop_total)*10000
gen Densid_ultraprocessados=( NumdeestabelUltra /pop_total)*10000
gen Densid_Mistros=( NumdeestabelMisto /pop_total)*10000

* Calculo da densidade, Populacao estimada 2016 (CAISAN)
gen Densid_innatura1=( NumdeestabelnNatura/NumHabitan)*10000
gen Densid_ultraprocessados1=( NumdeestabelUltra /NumHabitan)*10000
gen Densid_Mistros1=( NumdeestabelMisto /NumHabitan)*10000

*Categorizar as Densidades em Desertos
gen Ali_saud1= Densid_innatura1
gen Ali_saud2= Densid_Mistros1
gen Ali_Nsaud= Densid_ultraprocessados1
gen Ali_saud11=0
gen Ali_saud22=0
gen Ali_Nsaud1=0
replace Ali_saud11=1 if (Densid_innatura1<1.448645) 
replace Ali_saud22=1 if (Densid_Mistros1<10.5)  
replace Ali_Nsaud1=1 if (Densid_ultraprocessados1>=0.7)

*Como remover os missings na base

replace Var1=0 if var1==0  *subustiruir por o os missing
browse if cod_mun==.
Drop if cod_mun==.
sum var 1, var 2, var 3..., detail

*Mortalidade Geral

gen txmort_todas= ( MorteTodas / idade30a69)*10000
Label variable txmort_todas "Taxa de mortalidade,todas causas"


gen txmort_todas_fem=( SexoF / mulheres)*10000
label variable txmort_todas_fem "Taxa de mortalidade, todas as causas, sexo femenino"

gen txmort_todas_masc=( sexoM / homens)*10000
label variable txmort_todas_masc "Taxa de mortalidade, todas as causas, sexo masculino"

gen txmort_todas_30a39= ( Idade30a39 / a39anos)*10000
label variable txmort_todas_30a39 "Taxa de mortalidade,todas_30a39"


gen txmort_todas_40a49= (Idade40a49 *10000)/ a49anos
label variable txmort_todas_40a49 "Taxa de mortalidade,todas_40a49"

gen txmort_todas_50a59= ( Idade50a59 *10000)/ a59anos
label variable txmort_todas_50a59 "Taxa de mortalidade, todas_50a59"

gen txmort_todas_60a69= ( Idade60a69 *10000)/ a69anos
label variable txmort_todas_60a69 "Taxa de mortalidade, todas_60a69"

*Taxas todas por raca

gen txmort_todas_branco= ( RacaBranca *10000)/ Brancos_total
label variable  txmort_todas_branco "Taxa de mortalidade, todas branco"

gen txmort_todas_preta= (RacaPreta *10000)/ Pretos_total
label variable  txmort_todas_preta "Taxa de mortalidade, todas Preta"

gen txmort_todas_parda= ( RacaParda *10000)/ Pardos_total
label variable  txmort_todas_parda "Taxa de mortalidade, todas Parda"

gen txmort_todas_indigena= ( RacaIndigena *10000)/ Indiginas_total
label variable  txmort_todas_indigena "Taxa de mortalidade, todas Indigena"

gen txmort_todas_amarela= ( RacaAmarela *10000)/ Amarelo_total
label variable  txmort_todas_amarela "Taxa de mortalidade, todas amarela"



 *Calcular taxa de mortalidade por “Outras as causas” segundo sexo, idade e raça/cor"

gen txmort_outras= ( MorteOutras / idade30a69)*10000
Label variable txmort_Outras_30a69 "Taxa de mortalidade, outras causas"


gen txmort_outras_fem=( MorteOutras_SexoF / mulheres)*10000
label variable txmort_outras_fem "Taxa de mortalidade, outras causas, sexo femenino"

gen txmort_outras_masc=( MorteOutras_SexoM / homens)*10000
label variable txmort_outras_masc "Taxa de mortalidade, outras causas, sexo masculino"

gen txmort_outras_30a39= ( MorteOutras_Idade30a39 / a39anos)*10000
label variable txmort_outras_30a39 "Taxa de mortalidade,outras_30a39"


gen txmort_outras_40a49= (MorteOutras_Idade40a49 *10000)/ a49anos
label variable txmort_outras_40a49 "Taxa de mortalidade,outras_40a49"

gen txmort_outras_50a59= ( MorteOutras_Idade50a59 *10000)/ a59anos
label variable txmort_outras_50a59 "Taxa de mortalidade, outras_50a59"

gen txmort_outras_60a69= ( MorteOutras_Idade60a69 *10000)/ a69anos
label variable txmort_outras_60a69 "Taxa de mortalidade, outras_60a69"

*Taxas todas por raca

gen txmort_outras_branco= ( MorteOutras_RacaBranca *10000)/ Brancos_total
label variable  txmort_outras_branco "Taxa de mortalidade, outras branco"

gen txmort_outras_preta= (MorteOutras_RacaPreta *10000)/ Pretos_total
label variable  txmort_outras_preta "Taxa de mortalidade, outras Preta"

gen txmort_outras_parda= ( MorteOutras_RacaParda *10000)/ Pardos_total
label variable  txmort_outras_parda "Taxa de mortalidade, outras Parda"

gen txmort_outras_indigena= ( MorteOutras_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_outras_indigena "Taxa de mortalidade, outras Indigena"

gen txmort_outras_amarela= ( MorteOutras_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_outras_amarela "Taxa de mortalidade, outras amarela"

*Calcular taxa de mortalidade por “DCV” segundo sexo, idade e raça/cor"

gen txmort_dcv= ( MorteDCV / idade30a69)*10000
Label variable txmort_dcv "Taxa de mortalidade por dcv"


gen txmort_dcv_fem=( MorteDCV_SexoF / mulheres)*10000
label variable txmort_dcv_fem "Taxa de mortalidade por dcv sexo femenino"

gen txmort_dcv_masc=( MorteDCV_SexoM / homens)*10000
label variable txmort_dcv_masc "Taxa de mortalidade por dcv sexo masculino"

gen txmort_dcv_30a39= ( MorteDCV_Idade30a39 / a39anos)*10000
label variable txmort_dcv_30a39 "Taxa de mortalidade,dcv_30a39"

gen txmort_dcv_40a49= ( MorteDCV_Idade40a49 / a49anos)*10000
label variable txmort_dcv_40a49 "Taxa de mortalidade,dcv_40a49"

gen txmort_dcv_50a59= ( MorteDCV_Idade50a59 / a59anos)*10000
label variable txmort_dcv_50a59 "Taxa de mortalidade,dcv_50a59"

gen txmort_dcv_60a69= ( MorteDCV_Idade60a69 / a69anos)*10000
label variable txmort_dcv_60a69 "Taxa de mortalidade,dcv_60a69"

*Taxas todas por raca

gen txmort_dcv_branco= ( MorteDCV_RacaBranca *10000)/ Brancos_total
label variable  txmort_dcv_branco "Taxa de mortalidade, dcv branco"

gen txmort_dcv_preta= (MorteDCV_RacaPreta *10000)/ Pretos_total
label variable  txmort_dcv_preta "Taxa de mortalidade, dcv Preta"

gen txmort_dcv_parda= ( MorteDCV_RacaParda *10000)/ Pardos_total
label variable  txmort_dcv_parda "Taxa de mortalidade, dcv Parda"

gen txmort_dcv_indigena= ( MorteDCV_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_dcv_indigena "Taxa de mortalidade, dcv Indigena"

gen txmort_dcv_amarela= ( MorteDCV_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_dcv_amarela "Taxa de mortalidade, dcv amarela"


*Calcular taxa de mortalidade por “AVC” segundo sexo, idade e raça/cor"

gen txmort_avc= ( MorteAVC / idade30a69)*10000
Label variable txmort_avc "Taxa de mortalidade por avc"


gen txmort_avc_fem=( MorteAVC_SexoF / mulheres)*10000
label variable txmort_avc_fem "Taxa de mortalidade por avc sexo femenino"

gen txmort_avc_masc=( MorteAVC_SexoM / homens)*10000
label variable txmort_avc_masc "Taxa de mortalidade por avc sexo masculino"

gen txmort_avc_30a39= ( MorteAVC_Idade30a39 / a39anos)*10000
label variable txmort_avc_30a39 "Taxa de mortalidade,avc_30a39"

gen txmort_avc_40a49= ( MorteAVC_Idade40a49 / a49anos)*10000
label variable txmort_avc_40a49 "Taxa de mortalidade,avc_40a49"

gen txmort_avc_50a59= ( MorteAVC_Idade50a59 / a59anos)*10000
label variable txmort_avc_50a59 "Taxa de mortalidade,avc_50a59"

gen txmort_avc_60a69= ( MorteAVC_Idade60a69 / a69anos)*10000
label variable txmort_avc_60a69 "Taxa de mortalidade,avc_60a69"

*Taxas todas por raca

gen txmort_avc_branco= ( MorteAVC_RacaBranca *10000)/ Brancos_total
label variable  txmort_avc_branco "Taxa de mortalidade, dcv branco"

gen txmort_avc_preta= (MorteAVC_RacaPreta *10000)/ Pretos_total
label variable  txmort_avc_preta "Taxa de mortalidade, dcv Preta"

gen txmort_avc_parda= ( MorteAVC_RacaParda *10000)/ Pardos_total
label variable  txmort_avc_parda "Taxa de mortalidade, dcv Parda"

gen txmort_avc_indigena= ( MorteAVC_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_avc_indigena "Taxa de mortalidade, dcv Indigena"

gen txmort_avc_amarela= ( MorteAVC_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_avc_amarela "Taxa de mortalidade, dcv amarela"

* infarto 
*Calcular taxa de mortalidade por “Infarto” segundo sexo, idade e raça/cor"

gen txmort_Infarto= ( MorteInfarto / idade30a69)*10000
Label variable txmort_avc "Taxa de mortalidade por Infarto"


gen txmort_Infarto_fem=( MorteInfarto_SexoF / mulheres)*10000
label variable txmort_Infarto_fem "Taxa de mortalidade por Infarto sexo femenino"

gen txmort_Infarto_masc=( MorteInfarto_SexoM / homens)*10000
label variable txmort_Infarto_masc "Taxa de mortalidade por Infarto sexo masculino"

gen txmort_Infarto_30a39= ( MorteInfarto_Idade30a39 / a39anos)*10000
label variable txmort_Infarto_30a39 "Taxa de mortalidade,Infarto_30a39"

gen txmort_Infarto_40a49= ( MorteInfarto_Idade40a49 / a49anos)*10000
label variable txmort_Infarto_40a49 "Taxa de mortalidade,Infarto_40a49"

gen txmort_Infarto_50a59= ( MorteInfarto_Idade50a59 / a59anos)*10000
label variable txmort_Infarto_50a59 "Taxa de mortalidade,Infarto_50a59"

gen txmort_Infarto_60a69= ( MorteInfarto_Idade60a69 / a69anos)*10000
label variable txmort_Infarto_60a69 "Taxa de mortalidade,Infarto_60a69"

*Taxas todas por raca

gen txmort_Infarto_branco= ( MorteInfarto_RacaBranca *10000)/ Brancos_total
label variable  txmort_Infarto_branco "Taxa de mortalidade, Infarto branco"

gen txmort_Infarto_preta= (MorteInfarto_RacaPreta *10000)/ Pretos_total
label variable  txmort_Infarto_preta "Taxa de mortalidade, Infarto Preta"

gen txmort_Infarto_parda= ( MorteInfarto_RacaParda *10000)/ Pardos_total
label variable  txmort_Infarto_parda "Taxa de mortalidade, Infarto Parda"

gen txmort_Infarto_indigena= ( MorteInfarto_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_Infarto_indigena "Taxa de mortalidade, Infarto Indigena"

gen txmort_Infarto_amarela= ( MorteInfarto_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_Infarto_amarela "Taxa de mortalidade, Infarto amarela"


*PADRONIZANDO AS TAXAS 
*Mortalidade Geral

gen txmort_todasp= (txmort_todas*10236)/10000
Label variable txmort_todasp "Taxa p de mortalidade,todas causas"


gen txmort_todas_femp=( txmort_todas_fem*4875)/10000
label variable txmort_todas_femp "Taxa p de mortalidade, todas as causas, sexo femenino"

gen txmort_todas_mascp=(txmort_todas_masc*5360)/10000
label variable txmort_todas_mascp "Taxa p de mortalidade, todas as causas, sexo masculino"

gen txmort_todas_30a39p= (txmort_todas_30a39*3662)/10000
label variable txmort_todas_30a39p "Taxa p de mortalidade,todas_30a39"


gen txmort_todas_40a49p= (txmort_todas_40a49*3281)/10000 
label variable txmort_todas_40a49p "Taxa p de mortalidade,todas_40a49"

gen txmort_todas_50a59p= ( txmort_todas_50a59*2066)/10000
label variable txmort_todas_50a59p "Taxa p de mortalidade, todas_50a59"

gen txmort_todas_60a69p= ( txmort_todas_60a69*1227)/10000
label variable txmort_todas_60a69 "Taxa p de mortalidade, todas_60a69"

*Taxas todas por raca

gen txmort_todas_brancop= (txmort_todas_branco*4495)/10000
label variable  txmort_todas_brancop "Taxa p de mortalidade, todas branco"

gen txmort_todas_pretap= (txmort_todas_preta*598)/10000
label variable  txmort_todas_pretap "Taxa p de mortalidade, todas Preta"

gen txmort_todas_pardap= ( txmort_todas_parda*4988)/ 10000
label variable  txmort_todas_pardap "Taxa p de mortalidade, todas Parda"

gen txmort_todas_indigenap= ( txmort_todas_indigena*84)/ 10000
label variable  txmort_todas_indigenap "Taxa p de mortalidade, todas Indigena"

gen txmort_todas_amarelap= ( txmort_todas_amarela*71)/ 10000
label variable  txmort_todas_amarelap "Taxa p de mortalidade, todas amarela"



 *Calcular taxa de mortalidade por “Outras as causas” segundo sexo, idade e raça/cor"

gen txmort_outras= ( MorteOutras / idade30a69)*10000
Label variable txmort_Outras_30a69 "Taxa de mortalidade, outras causas"


gen txmort_outras_fem=( MorteOutras_SexoF / mulheres)*10000
label variable txmort_outras_fem "Taxa de mortalidade, outras causas, sexo femenino"

gen txmort_outras_masc=( MorteOutras_SexoM / homens)*10000
label variable txmort_outras_masc "Taxa de mortalidade, outras causas, sexo masculino"

gen txmort_outras_30a39= ( MorteOutras_Idade30a39 / a39anos)*10000
label variable txmort_outras_30a39 "Taxa de mortalidade,outras_30a39"


gen txmort_outras_40a49= (MorteOutras_Idade40a49 *10000)/ a49anos
label variable txmort_outras_40a49 "Taxa de mortalidade,outras_40a49"

gen txmort_outras_50a59= ( MorteOutras_Idade50a59 *10000)/ a59anos
label variable txmort_outras_50a59 "Taxa de mortalidade, outras_50a59"

gen txmort_outras_60a69= ( MorteOutras_Idade60a69 *10000)/ a69anos
label variable txmort_outras_60a69 "Taxa de mortalidade, outras_60a69"

*Taxas todas por raca

gen txmort_outras_branco= ( MorteOutras_RacaBranca *10000)/ Brancos_total
label variable  txmort_outras_branco "Taxa de mortalidade, outras branco"

gen txmort_outras_preta= (MorteOutras_RacaPreta *10000)/ Pretos_total
label variable  txmort_outras_preta "Taxa de mortalidade, outras Preta"

gen txmort_outras_parda= ( MorteOutras_RacaParda *10000)/ Pardos_total
label variable  txmort_outras_parda "Taxa de mortalidade, outras Parda"

gen txmort_outras_indigena= ( MorteOutras_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_outras_indigena "Taxa de mortalidade, outras Indigena"

gen txmort_outras_amarela= ( MorteOutras_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_outras_amarela "Taxa de mortalidade, outras amarela"

*Calcular taxa de mortalidade por “DCV” segundo sexo, idade e raça/cor"

gen txmort_dcv= ( MorteDCV / idade30a69)*10000
Label variable txmort_dcv "Taxa de mortalidade por dcv"


gen txmort_dcv_fem=( MorteDCV_SexoF / mulheres)*10000
label variable txmort_dcv_fem "Taxa de mortalidade por dcv sexo femenino"

gen txmort_dcv_masc=( MorteDCV_SexoM / homens)*10000
label variable txmort_dcv_masc "Taxa de mortalidade por dcv sexo masculino"

gen txmort_dcv_30a39= ( MorteDCV_Idade30a39 / a39anos)*10000
label variable txmort_dcv_30a39 "Taxa de mortalidade,dcv_30a39"

gen txmort_dcv_40a49= ( MorteDCV_Idade40a49 / a49anos)*10000
label variable txmort_dcv_40a49 "Taxa de mortalidade,dcv_40a49"

gen txmort_dcv_50a59= ( MorteDCV_Idade50a59 / a59anos)*10000
label variable txmort_dcv_50a59 "Taxa de mortalidade,dcv_50a59"

gen txmort_dcv_60a69= ( MorteDCV_Idade60a69 / a69anos)*10000
label variable txmort_dcv_60a69 "Taxa de mortalidade,dcv_60a69"

*Taxas todas por raca

gen txmort_dcv_branco= ( MorteDCV_RacaBranca *10000)/ Brancos_total
label variable  txmort_dcv_branco "Taxa de mortalidade, dcv branco"

gen txmort_dcv_preta= (MorteDCV_RacaPreta *10000)/ Pretos_total
label variable  txmort_dcv_preta "Taxa de mortalidade, dcv Preta"

gen txmort_dcv_parda= ( MorteDCV_RacaParda *10000)/ Pardos_total
label variable  txmort_dcv_parda "Taxa de mortalidade, dcv Parda"

gen txmort_dcv_indigena= ( MorteDCV_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_dcv_indigena "Taxa de mortalidade, dcv Indigena"

gen txmort_dcv_amarela= ( MorteDCV_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_dcv_amarela "Taxa de mortalidade, dcv amarela"


*Calcular taxa de mortalidade por “AVC” segundo sexo, idade e raça/cor"

gen txmort_avc= ( MorteAVC / idade30a69)*10000
Label variable txmort_avc "Taxa de mortalidade por avc"


gen txmort_avc_fem=( MorteAVC_SexoF / mulheres)*10000
label variable txmort_avc_fem "Taxa de mortalidade por avc sexo femenino"

gen txmort_avc_masc=( MorteAVC_SexoM / homens)*10000
label variable txmort_avc_masc "Taxa de mortalidade por avc sexo masculino"

gen txmort_avc_30a39= ( MorteAVC_Idade30a39 / a39anos)*10000
label variable txmort_avc_30a39 "Taxa de mortalidade,avc_30a39"

gen txmort_avc_40a49= ( MorteAVC_Idade40a49 / a49anos)*10000
label variable txmort_avc_40a49 "Taxa de mortalidade,avc_40a49"

gen txmort_avc_50a59= ( MorteAVC_Idade50a59 / a59anos)*10000
label variable txmort_avc_50a59 "Taxa de mortalidade,avc_50a59"

gen txmort_avc_60a69= ( MorteAVC_Idade60a69 / a69anos)*10000
label variable txmort_avc_60a69 "Taxa de mortalidade,avc_60a69"

*Taxas todas por raca

gen txmort_avc_branco= ( MorteAVC_RacaBranca *10000)/ Brancos_total
label variable  txmort_avc_branco "Taxa de mortalidade, dcv branco"

gen txmort_avc_preta= (MorteAVC_RacaPreta *10000)/ Pretos_total
label variable  txmort_avc_preta "Taxa de mortalidade, dcv Preta"

gen txmort_avc_parda= ( MorteAVC_RacaParda *10000)/ Pardos_total
label variable  txmort_avc_parda "Taxa de mortalidade, dcv Parda"

gen txmort_avc_indigena= ( MorteAVC_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_avc_indigena "Taxa de mortalidade, dcv Indigena"

gen txmort_avc_amarela= ( MorteAVC_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_avc_amarela "Taxa de mortalidade, dcv amarela"

* infarto 
*Calcular taxa de mortalidade por “Infarto” segundo sexo, idade e raça/cor"

gen txmort_Infarto= ( MorteInfarto / idade30a69)*10000
Label variable txmort_avc "Taxa de mortalidade por Infarto"


gen txmort_Infarto_fem=( MorteInfarto_SexoF / mulheres)*10000
label variable txmort_Infarto_fem "Taxa de mortalidade por Infarto sexo femenino"

gen txmort_Infarto_masc=( MorteInfarto_SexoM / homens)*10000
label variable txmort_Infarto_masc "Taxa de mortalidade por Infarto sexo masculino"

gen txmort_Infarto_30a39= ( MorteInfarto_Idade30a39 / a39anos)*10000
label variable txmort_Infarto_30a39 "Taxa de mortalidade,Infarto_30a39"

gen txmort_Infarto_40a49= ( MorteInfarto_Idade40a49 / a49anos)*10000
label variable txmort_Infarto_40a49 "Taxa de mortalidade,Infarto_40a49"

gen txmort_Infarto_50a59= ( MorteInfarto_Idade50a59 / a59anos)*10000
label variable txmort_Infarto_50a59 "Taxa de mortalidade,Infarto_50a59"

gen txmort_Infarto_60a69= ( MorteInfarto_Idade60a69 / a69anos)*10000
label variable txmort_Infarto_60a69 "Taxa de mortalidade,Infarto_60a69"

*Taxas todas por raca

gen txmort_Infarto_branco= ( MorteInfarto_RacaBranca *10000)/ Brancos_total
label variable  txmort_Infarto_branco "Taxa de mortalidade, Infarto branco"

gen txmort_Infarto_preta= (MorteInfarto_RacaPreta *10000)/ Pretos_total
label variable  txmort_Infarto_preta "Taxa de mortalidade, Infarto Preta"

gen txmort_Infarto_parda= ( MorteInfarto_RacaParda *10000)/ Pardos_total
label variable  txmort_Infarto_parda "Taxa de mortalidade, Infarto Parda"

gen txmort_Infarto_indigena= ( MorteInfarto_RacaIndigena *10000)/ Indiginas_total
label variable  txmort_Infarto_indigena "Taxa de mortalidade, Infarto Indigena"

gen txmort_Infarto_amarela= ( MorteInfarto_RacaAmarela *10000)/ Amarelo_total
label variable  txmort_Infarto_amarela "Taxa de mortalidade, Infarto amarela"


*dvC POR IDADE 
gen tpdcv_30a39= txmort_dcv_30a39 *3662/10000
gen tpdcv_40a49= txmort_dcv_40a49 *3281/10000
gen tpdcv_50a59= txmort_dcv_50a59 *2066/10000
gen tpdcv_60a69= txmort_dcv_60a69 *1227/10000































