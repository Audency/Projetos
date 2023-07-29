*###############################################################################

/*

*####### SELECIONANDO PASTA E PREPARANDO BANCO #######

cd "C:\Users\italo\Desktop\ETLSINASC" 				// inserir caminho para a pasta descompactada do SINASC



*####### CONVERTENDO E SALVANDO ARQUIVOS DE CSV PARA DTA #######

local variaveis_manter = "res_sigla_uf ano_nasc idademae escmae sexo racacor peso gravidez" // semanagestac está faltando antes de 2009!  /  def_gestacao tem em todos.

foreach UF in AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO {
	foreach ANO in 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 {
		import delimited ETLSINASC.DNRES_`UF'_`ANO'_t.csv, clear
		keep `variaveis_manter'
		save ETLSINASC.DNRES_`UF'_`ANO'_t.dta, replace
	}

	use ETLSINASC.DNRES_`UF'_2000_t.dta, replace
	
	foreach ANO in 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 {

		save ETLSINASC.DNRES_`UF'.dta, replace
		append using ETLSINASC.DNRES_`UF'_`ANO'_t.dta, force

		save ETLSINASC.DNRES_`UF'.dta, replace
		erase ETLSINASC.DNRES_`UF'_`ANO'_t.dta
	}

	erase ETLSINASC.DNRES_`UF'_2000_t.dta

}



*####### CRIANDO BANCO SEGUDO UNIDADE DE ANÁLISE (BRASIL E REGIÕES) #######


*### BRASIL ###
use "ETLSINASC.DNRES_AC.dta", replace
foreach UF in AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO {
	append using ETLSINASC.DNRES_`UF'.dta, force
	save "ETLSINASC.DNRES_[BRASIL].dta", replace	
}


*### REGIÕES ###

	
use "ETLSINASC.DNRES_[BRASIL].dta", clear

*### Norte ###
preserve
keep if res_sigla_uf == "AC" | res_sigla_uf == "AP" | res_sigla_uf == "AM" | res_sigla_uf == "PA" | res_sigla_uf == "RO" | res_sigla_uf == "RR" | res_sigla_uf == "TO"
save "ETLSINASC.DNRES_[NORTE].dta", replace	
restore


*### Nordeste ###
preserve
keep if res_sigla_uf == "AL" | res_sigla_uf == "BA" | res_sigla_uf == "CE" | res_sigla_uf == "MA" | res_sigla_uf == "PB" | res_sigla_uf == "PE" | res_sigla_uf == "PI" | res_sigla_uf == "RN" | res_sigla_uf == "SE"
save "ETLSINASC.DNRES_[NORDESTE].dta", replace	
restore


*### Sudeste ###
preserve
keep if res_sigla_uf == "ES" | res_sigla_uf == "MG" | res_sigla_uf == "RJ" | res_sigla_uf == "SP" 
save "ETLSINASC.DNRES_[SUDESTE].dta", replace	
restore

	
*### Sul ###
preserve
keep if res_sigla_uf == "PR" | res_sigla_uf == "RS" | res_sigla_uf == "SC"
save "ETLSINASC.DNRES_[SUL].dta", replace	
restore

	
*### Centro-Oeste ###
preserve
keep if res_sigla_uf == "DF" | res_sigla_uf == "GO" | res_sigla_uf == "MT" | res_sigla_uf == "MS"
save "ETLSINASC.DNRES_[CENTRO-OESTE].dta", replace	
restore





*####### CRIANDO E ACOMPANHANDO VARIÁVEL GIG #######

foreach TERRITORIO in BRASIL NORTE NORDESTE SUDESTE SUL CENTRO-OESTE {   
	use "ETLSINASC.DNRES_[`TERRITORIO'].dta", clear
		
	gen fxetaria = .
	replace fxetaria = 1 if idademae < 20
	replace fxetaria = 2 if idademae >=20 & idademae <=29
	replace fxetaria = 3 if idademae >=30 & idademae <.
	
	gen bpn=.
	replace bpn=0 if peso>=2500 & peso<.
	replace bpn=1 if peso< 2500
	
	save "ETLSINASC.DNRES_[`TERRITORIO']_novo.dta", replace

	display "####################"
	display ""
	display ""
	display "Prevalência de baixo peso ao nascer entre filhos de mulheres de todas as faixas etárias, segundo ano, `TERRITORIO'"
	table ano, stat(mean bpn) nformat(%9.4f)
	display ""
	display "Prevalência de baixo peso ao nascer entre filhos de mulheres menores de 20 anos, segundo ano, `TERRITORIO'"
	display ""
	table ano if fxetaria==1, stat(mean bpn) nformat(%9.4f)
	display "Prevalência de baixo peso ao nascer entre filhos de mulheres de 20 a 29 anos, segundo ano, `TERRITORIO'"
	display ""
	table ano if fxetaria==2, stat(mean bpn) nformat(%9.4f)
	display ""
	display "Prevalência de baixo peso ao nascer entre filhos de mulheres de 30 anos ou mais, segundo ano, `TERRITORIO'"
	table ano if fxetaria==3, stat(mean bpn) nformat(%9.4f)
}


*/


*#################################################################################


*####### SELECIONANDO PASTA E PREPARANDO BANCO #######

cd "C:\Users\italo\Google Drive\8. DOUTORADO\15_Outros projetos\Audencio_GIG"
use "20221023_ Banco BPN.dta", clear

keep if ano>=2010 & ano<=2020

order br_total n_total ne_total se_total s_total co_total br_fxetaria1 n_fxetaria1 ne_fxetaria1 se_fxetaria1 s_fxetaria1 co_fxetaria1 br_fxetaria2 n_fxetaria2 ne_fxetaria2 se_fxetaria2 s_fxetaria2 co_fxetaria2 br_fxetaria3 n_fxetaria3 ne_fxetaria3 se_fxetaria3 s_fxetaria3 co_fxetaria3


*####### DETERMINANDO TSSET #######

tsset ano


*####### FAZENDO GRÁFICOS DE LINHA #######
tsline br_total n_total ne_total se_total s_total co_total, name(ts_total, replace) ylabel(6(1)12, gmin gmax angle(0)) legend(label(1 "Brasil") label(2 "Norte") label(3 "Nordeste") label(4 "Sudeste") label(5 "Sul") label(6 "Centro-Oeste") rows(2)) ytitle("Nascidos vivos de baixo peso ao nascer (%)", size(small)) xtitle("Ano", size(small)) title("Baixo peso ao nascer (Mães de todas as idades)", size(medium)) graphregion(color(white)) bgcolor(white)

tsline br_fxetaria1 n_fxetaria1 ne_fxetaria1 se_fxetaria1 s_fxetaria1 co_fxetaria1, name(ts_fxetaria1, replace) ylabel(6(1)12, gmin gmax angle(0)) legend(label(1 "Brasil") label(2 "Norte") label(3 "Nordeste") label(4 "Sudeste") label(5 "Sul") label(6 "Centro-Oeste") rows(2)) ytitle("Nascidos vivos de baixo peso ao nascer (%)", size(small)) xtitle("Ano", size(small)) title("Baixo peso ao nascer (Mães < 20 anos de idade)", size(medium)) graphregion(color(white)) bgcolor(white)

tsline br_fxetaria2 n_fxetaria2 ne_fxetaria2 se_fxetaria2 s_fxetaria2 co_fxetaria2, name(ts_fxetaria2, replace) ylabel(6(1)12, gmin gmax angle(0)) legend(label(1 "Brasil") label(2 "Norte") label(3 "Nordeste") label(4 "Sudeste") label(5 "Sul") label(6 "Centro-Oeste") rows(2)) ytitle("Nascidos vivos de baixo peso ao nascer (%)", size(small)) xtitle("Ano", size(small)) title("Baixo peso ao nascer (Mães de 20 a 29 anos de idade)", size(medium)) graphregion(color(white)) bgcolor(white)

tsline br_fxetaria3 n_fxetaria3 ne_fxetaria3 se_fxetaria3 s_fxetaria3 co_fxetaria3, name(ts_fxetaria3, replace) ylabel(6(1)12, gmin gmax angle(0)) legend(label(1 "Brasil") label(2 "Norte") label(3 "Nordeste") label(4 "Sudeste") label(5 "Sul") label(6 "Centro-Oeste") rows(2)) ytitle("Nascidos vivos de baixo peso ao nascer (%)", size(small)) xtitle("Ano", size(small)) title("Baixo peso ao nascer (Mães ≥ 30 anos de idade)", size(medium)) graphregion(color(white)) bgcolor(white)



*####### CRIANDO VARIÁVEIS EM ESCALA LOGARÍTMICA #######

foreach var in br_total n_total ne_total se_total s_total co_total br_fxetaria1 n_fxetaria1 ne_fxetaria1 se_fxetaria1 s_fxetaria1 co_fxetaria1 br_fxetaria2 n_fxetaria2 ne_fxetaria2 se_fxetaria2 s_fxetaria2 co_fxetaria2 br_fxetaria3 n_fxetaria3 ne_fxetaria3 se_fxetaria3 s_fxetaria3 co_fxetaria3 {
	gen ln_`var' = ln(`var')
}


*####### FAZENDO REGRESSÃO DE PRAIS #######

// Total
foreach var in ln_br_total ln_n_total ln_ne_total ln_se_total ln_s_total ln_co_total {
	prais `var' ano, nolog
	display ""
	display "APC `var' (%)"
	display ( -1 + (exp(r(table)[1,1])) ) * 100
	display ""
	display "IC95% INFERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[5,1])) ) * 100
	display ""
	display "IC95% SUPERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[6,1])) ) * 100
	display ""
	display "p `var' (%)"
	display r(table)[4,1]
	
	display ""
	display "MATRIZ `var'"
	mat matriz = [( -1 + (exp(r(table)[1,1])) ) * 100,( -1 + (exp(r(table)[5,1])) ) * 100,( -1 + (exp(r(table)[6,1])) ) * 100,r(table)[4,1]]
	mat colnames matriz = APC INF_IC95% SUP_IC95% p
	mat list matriz, format(%9.3f)
}

// <20
foreach var in ln_br_fxetaria1 ln_n_fxetaria1 ln_ne_fxetaria1 ln_se_fxetaria1 ln_s_fxetaria1 ln_co_fxetaria1 {
	prais `var' ano, nolog
	display ""
	display "APC `var' (%)"
	display ( -1 + (exp(r(table)[1,1])) ) * 100
	display ""
	display "IC95% INFERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[5,1])) ) * 100
	display ""
	display "IC95% SUPERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[6,1])) ) * 100
	display ""
	display "p `var' (%)"
	display r(table)[4,1]
	
	display ""
	display "MATRIZ `var'"
	mat matriz = [( -1 + (exp(r(table)[1,1])) ) * 100,( -1 + (exp(r(table)[5,1])) ) * 100,( -1 + (exp(r(table)[6,1])) ) * 100,r(table)[4,1]]
	mat colnames matriz = APC INF_IC95% SUP_IC95% p
	mat list matriz, format(%9.3f)
}

// 20-29
foreach var in ln_br_fxetaria2 ln_n_fxetaria2 ln_ne_fxetaria2 ln_se_fxetaria2 ln_s_fxetaria2 ln_co_fxetaria2 {
	prais `var' ano, nolog
	display ""
	display "APC `var' (%)"
	display ( -1 + (exp(r(table)[1,1])) ) * 100
	display ""
	display "IC95% INFERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[5,1])) ) * 100
	display ""
	display "IC95% SUPERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[6,1])) ) * 100
	display ""
	display "p `var' (%)"
	display r(table)[4,1]
	
	display ""
	display "MATRIZ `var'"
	mat matriz = [( -1 + (exp(r(table)[1,1])) ) * 100,( -1 + (exp(r(table)[5,1])) ) * 100,( -1 + (exp(r(table)[6,1])) ) * 100,r(table)[4,1]]
	mat colnames matriz = APC INF_IC95% SUP_IC95% p
	mat list matriz, format(%9.3f)
}

// 30+
foreach var in ln_br_fxetaria3 ln_n_fxetaria3 ln_ne_fxetaria3 ln_se_fxetaria3 ln_s_fxetaria3 ln_co_fxetaria3 {
	prais `var' ano, nolog
	display ""
	display "APC `var' (%)"
	display ( -1 + (exp(r(table)[1,1])) ) * 100
	display ""
	display "IC95% INFERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[5,1])) ) * 100
	display ""
	display "IC95% SUPERIOR APC `var' (%)"
	display ( -1 + (exp(r(table)[6,1])) ) * 100
	display ""
	display "p `var' (%)"
	display r(table)[4,1]
	
	display ""
	display "MATRIZ `var'"
	mat matriz = [( -1 + (exp(r(table)[1,1])) ) * 100,( -1 + (exp(r(table)[5,1])) ) * 100,( -1 + (exp(r(table)[6,1])) ) * 100,r(table)[4,1]]
	mat colnames matriz = APC INF_IC95% SUP_IC95% p
	mat list matriz, format(%9.3f)
}




































	