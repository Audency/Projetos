*Estratificacao do Sexo pela categoria da idade 

COMPUTE SexoM_Idade30a39= SexoM*Idade30a39 . 
EXECUTE.

COMPUTE SexoM_Idade40a49= SexoM*Idade40a49 . 
EXECUTE.

COMPUTE SexoM_Idade50a59= SexoM*Idade50a59 . 
EXECUTE. 

COMPUTE SexoM_Idade60a69= SexoM*Idade60a69 . 
EXECUTE.

*Estratificacao do Sexo pela categoria da idade 

COMPUTE SexoF_Idade30a39= SexoF*Idade30a39 . 
EXECUTE.

COMPUTE SexoF_Idade40a49= SexoF*Idade40a49 . 
EXECUTE.

COMPUTE SexoF_Idade50a59= SexoF*Idade50a59 . 
EXECUTE.

COMPUTE SexoF_Idade60a69= SexoF*Idade60a69 . 
EXECUTE.

*Estratificacao da morte por DCV por sexo Sexo pela categoria da idade 

*Masculino

COMPUTE DCV_SexoM_Idade30a39= SexoM*Idade30a39*MorteDCV . 
EXECUTE.

COMPUTE DCV_SexoM_Idade40a49= SexoM*Idade40a49*MorteDCV . 
EXECUTE.

COMPUTE DCV_SexoM_Idade50a59= SexoM*Idade50a59*MorteDCV . 
EXECUTE.

COMPUTE DCV_SexoM_Idade60a69= SexoM*Idade60a69*MorteDCV . 
EXECUTE.

*Femenino 

COMPUTE DCV_SexoF_Idade30a39= SexoF*Idade30a39*MorteDCV . 
EXECUTE.

COMPUTE DCV_SexoF_Idade40a49= SexoF*Idade40a49*MorteDCV . 
EXECUTE.

COMPUTE DCV_SexoF_Idade50a59= SexoF*Idade50a59*MorteDCV . 
EXECUTE.

COMPUTE DCV_SexoF_Idade60a69= SexoF*Idade60a69*MorteDCV . 
EXECUTE.


*Estratificacao da morte por AVC por sexo Sexo pela categoria da idade 

*Masculino

COMPUTE AVC_SexoM_Idade30a39= SexoM*Idade30a39*MorteAVC . 
EXECUTE.

COMPUTE AVC_SexoM_Idade40a49= SexoM*Idade40a49*MorteAVC . 
EXECUTE.

COMPUTE AVC_SexoM_Idade50a59= SexoM*Idade50a59*MorteAVC . 
EXECUTE.

COMPUTE AVC_SexoM_Idade60a69= SexoM*Idade60a69*MorteAVC . 
EXECUTE.

*Femenino 

COMPUTE AVC_SexoF_Idade30a39= SexoF*Idade30a39*MorteAVC . 
EXECUTE.

COMPUTE AVC_SexoF_Idade40a49= SexoF*Idade40a49*MorteAVC . 
EXECUTE.

COMPUTE AVC_SexoF_Idade50a59= SexoF*Idade50a59*MorteAVC . 
EXECUTE.

COMPUTE AVC_SexoF_Idade60a69= SexoF*Idade60a69*MorteAVC . 
EXECUTE.

*Estratificacao da morte por Infarto por sexo Sexo pela categoria da idade 

*Masculino

COMPUTE Infarto_SexoM_Idade30a39= SexoM*Idade30a39*MorteInfarto . 
EXECUTE.

COMPUTE Infarto_SexoM_Idade40a49= SexoM*Idade40a49*MorteInfarto . 
EXECUTE.

COMPUTE Infarto_SexoM_Idade50a59= SexoM*Idade50a59*MorteInfarto . 
EXECUTE.

COMPUTE Infarto_SexoM_Idade60a69= SexoM*Idade60a69*MorteInfarto . 
EXECUTE.

*Femenino 

COMPUTE Infarto_SexoF_Idade30a39= SexoF*Idade30a39*MorteInfarto . 
EXECUTE.

COMPUTE Infarto_SexoF_Idade40a49= SexoF*Idade40a49*MorteInfarto . 
EXECUTE.

COMPUTE Infarto_SexoF_Idade50a59= SexoF*Idade50a59*MorteInfarto . 
EXECUTE.

COMPUTE Infarto_SexoF_Idade60a69= SexoF*Idade60a69*MorteInfarto . 
EXECUTE.

*Estratificacao da Raca pela categoria da idade
 
*Branca

COMPUTE Branca_30a39=  RacaBranca*Idade30a39. 
EXECUTE.

COMPUTE Branca_40a49=  RacaBranca*Idade40a49. 
EXECUTE.

COMPUTE Branca_50a59= RacaBranca*Idade50a59. 
EXECUTE.

COMPUTE Branca_60a69= RacaBranca*Idade60a69. 
EXECUTE.

*Preto 

COMPUTE Preta_30a39= RacaPreta*Idade30a39. 
EXECUTE.

COMPUTE Preta_40a49= RacaPreta*Idade40a49. 
EXECUTE.

COMPUTE Preta_50a59= RacaPreta*Idade50a59. 
EXECUTE.

COMPUTE Preta_60a69= RacaPreta*Idade60a69. 

*Amarela
 
COMPUTE Amarela_30a39= RacaAmarela*Idade30a39. 
EXECUTE.

COMPUTE Amarela_40a49= RacaAmarela*Idade40a49. 
EXECUTE.

COMPUTE Amarela_50a59= RacaAmarela*Idade50a59. 
EXECUTE.

COMPUTE Amarela_60a69= RacaAmarela*Idade60a69. 

*Parda

COMPUTE Parda_30a39= RacaParda*Idade30a39. 
EXECUTE.

COMPUTE Parda_40a49= RacaParda*Idade40a49. 
EXECUTE.

COMPUTE Parda_50a59= RacaParda*Idade50a59. 
EXECUTE.

COMPUTE Parda_60a69= RacaParda*Idade60a69. 

*Indigena 

COMPUTE Indigena_30a39= RacaIndigena*Idade30a39. 
EXECUTE.

COMPUTE Indigena_40a49= RacaIndigena*Idade40a49. 
EXECUTE.

COMPUTE Indigena_50a59= RacaIndigena*Idade50a59. 
EXECUTE.

COMPUTE Indigena_60a69= RacaIndigena*Idade60a69. 


*Estratificacao da morte por DCV raca  pela categoria da idade 

*Branco 

COMPUTE DCV_Branca_30a39= MorteDCV * RacaBranca*Idade30a39. 
EXECUTE.

COMPUTE DCV_Branca_40a49= MorteDCV * RacaBranca*Idade40a49. 
EXECUTE.

COMPUTE DCV_Branca_50a59= MorteDCV * RacaBranca*Idade50a59. 
EXECUTE.

COMPUTE DCV_Branca_60a69= MorteDCV * RacaBranca*Idade60a69. 
EXECUTE.

*Preto 

COMPUTE DCV_Preta_30a39= MorteDCV * RacaPreta*Idade30a39. 
EXECUTE.

COMPUTE DCV_Preta_40a49= MorteDCV * RacaPreta*Idade40a49. 
EXECUTE.

COMPUTE DCV_Preta_50a59= MorteDCV * RacaPreta*Idade50a59. 
EXECUTE.

COMPUTE DCV_Preta_60a69= MorteDCV * RacaPreta*Idade60a69. 
EXECUTE.

*Amarela 

COMPUTE DCV_Amarela_30a39= MorteDCV * RacaAmarela *Idade30a39. 
EXECUTE.

COMPUTE DCV_Amarela_40a49= MorteDCV * RacaAmarela *Idade40a49. 
EXECUTE.

COMPUTE DCV_Amarela_50a59= MorteDCV * RacaAmarela *Idade50a59. 
EXECUTE.

COMPUTE DCV_Amarela_60a69= MorteDCV * RacaAmarela*Idade60a69. 

*Parda
 
COMPUTE DCV_Parda_30a39= MorteDCV * RacaParda *Idade30a39. 
EXECUTE.

COMPUTE DCV_Parda_40a49= MorteDCV * RacaParda *Idade40a49. 
EXECUTE.

COMPUTE DCV_Parda_50a59= MorteDCV * RacaParda *Idade50a59. 
EXECUTE.

COMPUTE DCV_Parda_60a69= MorteDCV * RacaParda*Idade60a69. 


*Indigena 

COMPUTE DCV_Indigena_30a39= MorteDCV * RacaIndigena *Idade30a39. 
EXECUTE.

COMPUTE DCV_Indigena_40a49= MorteDCV * RacaIndigena *Idade40a49. 
EXECUTE.

COMPUTE DCV_Indigena_50a59= MorteDCV * RacaIndigena *Idade50a59. 
EXECUTE.

COMPUTE DCV_Indigena_60a69= MorteDCV * RacaIndigena*Idade60a69. 

*
*Estratificacao da morte por AVC raca  pela categoria da idade 

*Branco 

COMPUTE AVC_Branca_30a39= MorteAVC * RacaBranca*Idade30a39. 
EXECUTE.

COMPUTE AVC_Branca_40a49= MorteAVC * RacaBranca*Idade40a49. 
EXECUTE.

COMPUTE AVC_Branca_50a59= MorteAVC * RacaBranca*Idade50a59. 
EXECUTE.

COMPUTE AVC_Branca_60a69= MorteAVC * RacaBranca*Idade60a69. 
EXECUTE.

*Preto 

COMPUTE AVC_Preta_30a39= MorteAVC * RacaPreta*Idade30a39. 
EXECUTE.

COMPUTE AVC_Preta_40a49= MorteAVC * RacaPreta*Idade40a49. 
EXECUTE.

COMPUTE AVC_Preta_50a59= MorteAVC * RacaPreta*Idade50a59. 
EXECUTE.

COMPUTE AVC_Preta_60a69= MorteAVC * RacaPreta*Idade60a69. 
EXECUTE.

*Amarela 

COMPUTE AVC_Amarela_30a39= MorteAVC * RacaAmarela *Idade30a39. 
EXECUTE.

COMPUTE AVC_Amarela_40a49= MorteAVC * RacaAmarela *Idade40a49. 
EXECUTE.

COMPUTE AVC_Amarela_50a59= MorteAVC * RacaAmarela *Idade50a59. 
EXECUTE.

COMPUTE AVC_Amarela_60a69= MorteAVC * RacaAmarela*Idade60a69. 

*Parda
 
COMPUTE AVC_Parda_30a39= MorteAVC * RacaParda *Idade30a39. 
EXECUTE.

COMPUTE AVC_Parda_40a49= MorteAVC * RacaParda *Idade40a49. 
EXECUTE.

COMPUTE AVC_Parda_50a59= MorteAVC * RacaParda *Idade50a59. 
EXECUTE.

COMPUTE AVC_Parda_60a69= MorteAVC * RacaParda*Idade60a69. 


*Indigena 

COMPUTE AVC_Indigena_30a39= MorteAVC * RacaIndigena *Idade30a39. 
EXECUTE.

COMPUTE AVC_Indigena_40a49= MorteAVC * RacaIndigena *Idade40a49. 
EXECUTE.

COMPUTE AVC_Indigena_50a59= MorteAVC * RacaIndigena *Idade50a59. 
EXECUTE.

COMPUTE AVC_Indigena_60a69= MorteAVC * RacaIndigena*Idade60a69. 



COMPUTE MorteDCV_SexoF1=MorteDCV * SexoF. 
EXECUTE.


COMPUTE MorteInfarto_RacaBranca1= MorteInfarto * RacaBranca. 
EXECUTE.


COMPUTE MorteInfarto_RacaPreta1= MorteInfarto * RacaPreta. 
EXECUTE.


COMPUTE MorteInfarto_RacaParda1= MorteInfarto * RacaParda. 
EXECUTE.

COMPUTE MorteInfarto_RacaIndigena1= MorteInfarto * RacaIndigena. 
EXECUTE.

COMPUTE MorteInfarto_RacaAmarela1= MorteInfarto * RacaAmarela. 
EXECUTE.


COMPUTE MorteDCV_RacaBranca1= MorteDCV * RacaBranca. 
EXECUTE.


COMPUTE MorteDCV_RacaPreta1= MorteDCV * RacaPreta. 
EXECUTE.


COMPUTE MorteDCV_RacaParda1= MorteDCV * RacaParda. 
EXECUTE.

COMPUTE MorteDCV_RacaIndigena1= MorteDCV * RacaIndigena. 
EXECUTE.

COMPUTE MorteDCV_RacaAmarela1= MorteDCV * RacaAmarela. 
EXECUTE.


COMPUTE MorteAVC_RacaBranca1=MorteAVC * RacaBranca. 
EXECUTE.


COMPUTE MorteAVC_RacaPreta1= MorteAVC * RacaPreta. 
EXECUTE.


COMPUTE MorteAVC_RacaParda1= MorteAVC * RacaParda. 
EXECUTE.

COMPUTE MorteAVC_RacaIndigena1= MorteAVC * RacaIndigena. 
EXECUTE.

COMPUTE MorteAVC_RacaAmarela1= MorteAVC * RacaAmarela. 
EXECUTE.

