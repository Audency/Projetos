
* EXEMPLO PARA INTER/EXTRAPOLACAO DA VARIAVEL "POP_TOTAL"

* Nesse momento, voce precisa ter na mema base os dados dos censos 2000 e 2010, ou seja, as variaveis "pop_total2000" e "pop_total2010"

* criar variavel pop_total para o ano com dado faltante que se deseja inter/extrapolar
gen pop_total2016=.

    * OU caso tenha interesse em também inte-extrapolar os dados para TODOS os anos com valores faltantes, basta criar a variavel pop_total para cada um desses anos
	  gen pop_total2001=.
	  gen pop_total2002=.
	  gen pop_total2003=.
	  gen pop_total2004=.
	  gen pop_total2005=.
	  gen pop_total2006=.
	  gen pop_total2007=.
	  gen pop_total2008=.
	  gen pop_total2009=.
	  	 
	  gen pop_total2011=.
	  gen pop_total2012=.
	  gen pop_total2013=.
	  gen pop_total2014=.
	  gen pop_total2015=.
	  gen pop_total2016=.

* transformar o dataset de wide para long, permitindo a criacao da var pop_total e sua subsequente inter/extrapolacao para os anos com valores faltantes
reshape wide pop_total, i(cod_mun) j(ano)
reshape wide pop_total  pop_rural  pop_urbana  popporte  idhm  idhm_renda  idhm_longevidade  idhm_educacao  pib  pibpcapita  renda  renda_gini  analfa_total  analfa_mulher  esc_anos  desemprego  pobres  extr_pobres, i( ibge_code ) j( year )

reshape long pop_total, i(cod_mun) j(ano)



* inter/extrapolar valores para os anos com dados faltantes
sort ibge_code
by ibge_code: ipolate pop_total ano, gen(pop_total_epolate) epolate
by ibge_code: ipolate pop_rural ano, gen(pop_rural_epolate) epolate
by ibge_code: ipolate pop_urbana ano, gen(pop_urbana_epolate) epolate
by ibge_code: ipolate popporte ano, gen(popporte_epolate) epolate
by ibge_code: ipolate idhm ano, gen(idhm_epolate) epolate
by ibge_code: ipolate idhm_renda ano, gen(idhm_renda_epolate) epolate
by ibge_code: ipolate idhm_longevidade ano, gen(idhm_longevidade_epolate) epolate
by ibge_code: ipolate idhm_educacao ano, gen(idhm_educacao_epolate) epolate
by ibge_code: ipolate pibpcapita ano, gen(pibpcapita_epolate) epolate
by ibge_code: ipolate renda ano, gen(renda_epolate) epolate
by ibge_code: ipolate renda_gini ano, gen(renda_gini_epolate) epolate
by ibge_code: ipolate analfa_total ano, gen(analfa_total_epolate) epolate
by ibge_code: ipolate analfa_mulher ano, gen(analfa_mulher_epolate) epolate
by ibge_code: ipolate esc_anos ano, gen(esc_anos_epolate) epolate
by ibge_code: ipolate desemprego ano, gen(desemprego_epolate) epolate
by ibge_code: ipolate pobres ano, gen(pobres_epolate) epolate
by ibge_code: ipolate extr_pobres ano, gen(extr_pobres_epolate) epolate


* arrendodar os valores da variavel pop_total
gen pop_total_round= ceil(pop_total)

* Gerar novar variaveis do ano 2016
gen pop_total2016=.
gen pop_rural2016=.  
gen pop_urbana2016=.  
gen popporte2016=.  
gen idhm2016=.
gen idhm_renda2016=.  
gen idhm_longevidade2016=.  
gen idhm_educacao2016=.  
gen pib2016=.  
gen pibpcapita2016=.
gen renda2016=. 
gen renda_gini2016=.  
gen analfa_total2016=.  
gen analfa_mulher2016=.  
gen esc_anos2016=.  
gen desemprego2016=.  
gen pobres2016=.  
gen extr_pobres2016=.

*Convertendo a variavel em string
tostring popporte2016, replace


