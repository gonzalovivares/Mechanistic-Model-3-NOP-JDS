# Import packages that are needed for the simulation
using DiffEqGPU, CUDA, OrdinaryDiffEq, StaticArrays, TimerOutputs, CSV, Plots, Metrics, DataFrames, Statistics, Metaheuristics

############################
# Definition of model inputs
############################
feed = 21000                # Dry matter intake, g/d
ndf_intake = feed * 0.35    # NDF intake, g/d
st_intake = feed * 0.20     # Starch intake, g/d
wr_intake = feed * 0.10     # Soluble carbohydrates intake, g/d
p_intake = feed * 0.15      # Protein intake, g/d
la_intake =  feed * 0.035   # Lactic acid intake, g/d
missfr_intake = feed * 0.10 # Undetected fraction intake (soluble fibers), g/d           
pcell = 0.60                # Proportion of cellulose in NDF, -

ndf_dig = 0.60              # Rumen NDF digestibility , (0-1)
st_dig =  0.90              # Rumen starch digestibility , (0-1)
wr_dig = 0.97               # Rumen soluble carbohydrates digestibility , (0-1)
p_dig = 0.60                # Rumen protein digestibility , (0-1)
pH = 6.2                    # Rumen pH
FrAc = 5.8                  # Dietary concentration of acetate, g/kg DM
ffmCe = 0.59                # Fraction of fermented hexoses employed on VFA production, and not on microbial biomass generation, calculated by Dijkstra et al. (1992) https://doi.org/10.1093/jn/122.11.2239
micr_n_fermented = 3.0      # Microbial nitrogen fermented in the rumen, mol/d, calculated by Dijkstra et al. (1992) https://doi.org/10.1093/jn/122.11.2239
UHaMMa = 2.20               # Fermentation of microbial hexoses from microbial recycling in the rumen, mol/d; calculated by Dijkstra et al. (1992) https://doi.org/10.1093/jn/122.11.2239
QMa =  400.0                # Pool of amylolytic microbes, calculated by Dijkstra et al. (1992) https://doi.org/10.1093/jn/122.11.2239
QMc =  1600.0               # Pool of fibrolytic microbes, calculated by Dijkstra et al. (1992) https://doi.org/10.1093/jn/122.11.2239
Fr3NOP = 60.0               # Dietary concentration of 3-nitrooxypropanol in the rumen, mg/kg DM

model_inputs_list = [feed, ndf_intake, st_intake, wr_intake, p_intake, la_intake, missfr_intake, pcell, 
                ndf_dig, st_dig, wr_dig, p_dig, pH,  FrAc,
                ffmCe, micr_n_fermented, UHaMMa, QMa, QMc, Fr3NOP]


inputs_df = DataFrame(model_inputs_list', :auto)
inputs_df = vcat(inputs_df, inputs_df, inputs_df, inputs_df, inputs_df)
inputs_df[: , end] = [0.0001, 20, 40, 60, 80]


######################################## 
# Functions defining the model equationskRCOxH2
########################################
function model_eqs_3NOP_model( u, p, t)

    # Define the pools
    QRCred = u[1]       
    QRCox = u[2]
    QHy = u[3]
    QMe = u[4]
    QAc = u[5]
    QPr = u[6]
    QBu = u[7]
    QVl = u[8]
    Q3NOP = u[9]
    Qeth = u[10]
    QMCRactive = u[11]
    QMCRinact = u[12]

    pRCred = QRCred / (QRCred + QRCox)                      # Proportion of redox cofactors in reduced state
    pMCRactive =  QMCRactive / (QMCRactive + QMCRinact)     # Proportion of MCR enzyme in active state

    # MODEL INPUTS UNFOLDING
    feed, ndf_intake, st_intake, wr_intake, p_intake, la_intake, missfr_intake, pcell, 
    ndf_dig, st_dig, wr_dig, p_dig, pH,  FrAc,
    ffmCe, micr_n_fermented, UHaMMa, QMa, QMc, Fr3NOP = p


    ##################
    # Model parameters 
    ##################
    
    # Parameters VFA stoichiometry, for reference read Vivares et al (2026a) https://doi.org/10.3168/jds.2025-28000
    Y0AcW = 0.45
    Y0AcS = 0.54
    Y0AcH = 0.66
    Y0AcC = 0.73
    Y0AcP = 0.57

    Y0PrW = 0.12
    Y0PrS = 0.15
    Y0PrH = 0.08
    Y0PrC = 0.13
    Y0PrP = 0.26
    
    Y0BuW = 0.40
    Y0BuS = 0.21
    Y0BuH = 0.26
    Y0BuC = 0.13
    Y0BuP = 0.10
        
    Y0VlW = 0.03
    Y0VlS = 0.10
    Y0VlH = 0.00
    Y0VlC = 0.01
    Y0VlP = 0.07
    
    diffyacW = -0.20
    diffyprW = 0.04
    diffybuW = 0.12
    diffYVlW = 0.04
    mpRCredW = 0.38
    opRCredW = 6.9

    diffyacS = -0.17
    diffyprS = 0.06
    diffybuS = -0.01
    diffYVlS = 0.12
    mpRCredS = 0.32
    opRCredS = 7.0
    
    diffacW_ph = -0.22
    diffprW_ph = 0.32
    diffbuW_ph = - (diffacW_ph + diffprW_ph)
    mphacW = 5.68
    ophacW = 16.6
   
    diffacS_ph = -0.36
    diffprS_ph = 0.32
    diffbuS_ph = - (diffacS_ph + diffprS_ph)
    mphacS = 5.75
    ophacS = 27.3
    
    phem = 1.0f0 - pcell

    fS_NSC = (st_intake * st_dig) / (st_intake * st_dig + wr_intake * wr_dig) 
    fW_NSC = (wr_intake * wr_dig) / (st_intake * st_dig + wr_intake * wr_dig) 

    YRCredAc = 2.0f0
    YRCredBu = 2.0f0
    RRCredPr = 1.0f0
    RRCredVl = 2.0f0

    kRCOxH2 = 2.48f04
    vhych4 = 3.64
    mhyhych4 = 1.31f-5
    k_evap = 100.2

    jdmiMeFl = 30.4 
    minMeFl = 0.13
    
    # PARAMETERS RELATED TO 3-NITROOXYPROPANOL
    V3NOPAb = 6.4               # Max. fractional rate of 3NOP absorption
    M3NOPAb = 1.4f-4            # MichaelisMenten affinity of 3NOP  for 3NOP absorption
    m3nopma = 0.014 
    
    jhyMePa = 6.8f-4
    v3nopMCR = 0.011
    m3nopMCRinact = 3.7f-05  
    kMCRreact = 4.3
    jhy3nopMe = 2.8f-4
    
    mh2hexeth = 1.9f-03
    oh2hexeth = 1.9
    kethbu  = 9.1
    kEthAb = 16.8

    khyac = 858.0
    mhyhyac = 2.4f-03

    YATP_0 = 6.0 
    diff_atp = 0.52
    m_chyme_atp = 1.4f-03
    o_chyme_atp = 2.9

    vru = 47.86f0 + (1.759f0 * (feed/1000.0f0))     # L, Rumen volume
    kFlPa = 0.97f0 + (0.116f0 * (feed / 1000.0f0))  # /d, fluid passage rate
    kSoPa = 0.57f0 + (0.017f0 * (feed / 1000.0f0))  # /d, solid passage rate
    
    #######################################
    # Concentrations = Pool / Rumen volume
    #######################################
    CHy = QHy / vru
    CAc = QAc / vru
    CPr = QPr / vru
    CBu = QBu / vru
    CVl = QVl / vru
    C3NOP = Q3NOP / vru
    Ceth = Qeth / vru


    ###########################
    # Pool of acetate, QAc, mol
    ###########################
    ### Production rates
    YAcC = Y0AcC  
    YAcH = Y0AcH  
    YAcS = Y0AcS + diffyacS / (1.0f0 +  (mpRCredS / pRCred)^opRCredS)  + (diffacS_ph / (1.0f0 + (pH / mphacS)^ophacS)   )
    YAcW = Y0AcW + diffyacW / (1.0f0 +  (mpRCredW / pRCred)^opRCredW)  + (diffacW_ph / (1.0f0 +  (pH / mphacW)^ophacW)   )
    YAcP = Y0AcP 

    # Production of Acetate from cellulose fermentation, mol/d
    PAcC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe * YAcC

    # Production of Acetate from hemicellulose fermentation, mol/d
    PAcH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe * YAcH

    # Production of Acetate from Starch fermentation, mol/d
    PAcS = ((st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe *  YAcS ) + ( UHaMMa * 0.711 * YAcS * fS_NSC)

    # Production of Acetate from soluble carbohydrates fermentation, mol/d
    PAcW = (((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe *  YAcW ) + (UHaMMa * 0.711 * fW_NSC * YAcW)

    # Production of Acetate from protein fermentation, mol/d
    PAcP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented )* (YAcP)   

    

    # Total Acetate production from feed fermentation, mol/d
    YCHAc = 2.0f0 - 1.0f0 / (1.0f0 + (mh2hexeth / CHy) ^ oh2hexeth )
    PAc = (PAcC + PAcH + PAcS + PAcW) * YCHAc + (PAcP) * 1.07f0

    # Acetate production from reductive acetogenesis, mol/d
    UH2Ac = QHy * khyac / (1.0f0 + (mhyhyac / CHy) )
    PAcH2Ac = UH2Ac / 4.0f0

    # Acetate intake from feed (silage mostly)
    yacfac = 1.0f0 / 60.0f0
    Ac_intake = FrAc * feed / 1000.0f0 * yacfac

    ### Uptake rates
    
    # Uptake for absorption, mol/d  (Parameters derived from Dijkstra et al. 1992)      
    UAcCAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QAc / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UAcCPa = kFlPa * QAc
    # Uptake for butyrate together with ethanol, mol/d    
    UEthBu = Qeth * kethbu 
    UAcEthBu = UEthBu * 0.6
    

    # Differential equation

    QAc_dt = PAc + Ac_intake - UAcCAb - UAcCPa + PAcH2Ac - UAcEthBu

    ##############################
    # Pool of propionate, QPr, mol
    ##############################
    # Inputs
    YPrC = Y0PrC 
    YPrH = Y0PrH  
    YPrS = Y0PrS + diffyprS / (1.0f0 +  (mpRCredS / pRCred)^opRCredS)  + (diffprS_ph / (1.0f0 +  (pH / mphacS) ^ ophacS)  )
    YPrW = Y0PrW + diffyprW / (1.0f0 +  (mpRCredW / pRCred)^opRCredW) + diffprW_ph / (1.0f0 +  (pH / mphacW) ^ ophacW)
    YPrP = Y0PrP

    ### Production rates

    # Production of Propionate from cellulose fermentation, mol/d
    PPrC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe * YPrC  
    # Production of Propionate from hemicellulose fermentation, mol/d
    PPrH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe * YPrH  
    # Production of Propionate from Starch fermentation, mol/d
    PPrS = (st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe *  YPrS + ( UHaMMa * 0.711 * YPrS *fS_NSC)
    # Production of Propionate from soluble carbohydrates fermentation, mol/d
    PPrW = ((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe * YPrW  + (UHaMMa * 0.711 * fW_NSC * YPrW)
    # Production of Propionate from protein fermentation, mol/d
    PPrP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented )* ( YPrP)  
    # Total Propionate production from feed fermentation, mol/d
    PPr = (PPrC + PPrH + PPrS + PPrW) * 2.0f0 + (PPrP) * 1.07f0

    ### Uptake rates
    
    # Uptake for absorption, mol/d  (Parameters derived from Dijkstra et al. 1992)      
    UPrPAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QPr / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UPrPPa = kFlPa * QPr

    # Differential equation
    QPr_dt = PPr - UPrPAb - UPrPPa

    ############################ 
    # Pool of butyrate, QBu, mol
    ############################ 
    # Inputs
    YBuC = Y0BuC  
    YBuH = Y0BuH  
    YBuS = Y0BuS + diffybuS / (1.0f0 +  (mpRCredS / pRCred) ^ opRCredS) + (diffbuS_ph / (1.0f0 +  (pH / mphacS) ^ ophacS)  )
    YBuW =  Y0BuW + diffybuW / (1.0f0 +  (mpRCredW / pRCred) ^ opRCredW) + (diffbuW_ph / (1.0f0 +  (pH / mphacW) ^ ophacW)  )
    YBuP = Y0BuP

    ### Production rates

    # Production of Butyrate from cellulose fermentation, mol/d
    PBuC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe *  YBuC
    # Production of Butyrate from hemicellulose fermentation, mol/d
    PBuH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe *  YBuH
    # Production of Butyrate from Starch fermentation, mol/d
    PBuS = (st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe * YBuS + ( UHaMMa * 0.711 * YBuS *fS_NSC)
    # Production of Butyrate from soluble carbohydrates fermentation, mol/d
    PBuW = ((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe *  YBuW  + (UHaMMa * 0.711 * fW_NSC * YBuW)
    # Production of Butyrate from protein fermentation, mol/d
    PBuP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented )*   YBuP

    # Total Butyrate production from feed fermentation, mol/d
    PBu = PBuC + PBuH + PBuS + PBuW + PBuP * 0.535f0
    # Production of butyrate from ethanol and acetate, mol/d
    PBuEthAc = UEthBu * 0.8

    ### Uptake rates
    
    # Uptake for absorption, mol/d (Parameters derived from Dijkstra et al. 1992)   
    UBuBAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QBu / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UBuBPa = kFlPa * QBu

    # Differential equation
    QBu_dt = PBu - UBuBAb - UBuBPa + PBuEthAc

    ################################################
    # Pool of valerate and other minor VFA, QVl, mol
    ################################################ 
    # Inputs
    YVlC = Y0VlC  
    YVlH = Y0VlH  
    YVlS = Y0VlS + diffYVlS / (1.0f0 +  (mpRCredS / pRCred) ^ opRCredS)
    YVlW = Y0VlW + diffYVlW / (1.0f0 +  (mpRCredW / pRCred) ^ opRCredW)
    YVlP = Y0VlP  

    ### Production rates

    # Production of Valerate and other minor VFA from cellulose fermentation, mol/d
    PVlC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe *  YVlC
    # Production of Valerate and other minor VFA from hemicellulose fermentation, mol/d
    PVlH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe *  YVlH
    # Production of Valerate and other minor VFA from Starch fermentation, mol/d
    PVlS = (st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe *  YVlS + ( UHaMMa * 0.711 * YVlS * fS_NSC)
    # Production of Valerate and other minor VFA from soluble carbohydrates fermentation, mol/d
    PVlW = ((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe * YVlW  + (UHaMMa * 0.711 * fW_NSC * YVlW)
    # Production of Valerate and other minor VFA from protein fermentation, mol/d
    PVlP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented ) * YVlP

    # Total Valerate and other minor VFA production from feed fermentation, mol/d
    PVl = PVlC + PVlH + PVlS + PVlW + PVlP * 0.535f0

    ### Uptake rates
    
    # Uptake for absorption, mol/d  (Parameters derived from Dijkstra et al. 1992)     
    UVlVAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QVl / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UVlVPa = kFlPa * QVl

    ### Differential equation
    QVl_dt = PVl - UVlVAb - UVlVPa


    ####################################################################### 
    # Pool of redox cofactors in the reduced state (i.e., NADH, Fdred), mol
    #######################################################################

    ### Production rates
    
    PRCredAcRCred = YRCredAc * PAc
    
    ### Uptake rates
    UEthBu = Qeth * kethbu 
    PBuEthAc = UEthBu * 0.8
    PRCredBuRCred = YRCredBu * PBu + PBuEthAc * 0.5

    # Uptake fluxes
    URCredRCredPr = RRCredPr * PPr
    URCredRCredVl = RRCredVl * PVl
    
    mhy_ft = 0.0005f0 + 0.0005f0 * pRCred
    Ft = 1.0f0 / (1.0f0 + ( CHy /mhy_ft)^3.0f0)
    URCredHy = QRCred * kRCOxH2 * Ft
    
    ### Differential equation
    QRCred_dt =  PRCredAcRCred + PRCredBuRCred - URCredRCredPr - URCredRCredVl - URCredHy

    #######################################################################
    # Pool of redox cofactors in the oxidized state (i.e., NAD, Fdox), mol
    #######################################################################
    ### Differential equation --> Inverse signs to QRCred_dt
    QRCox_dt = - PRCredAcRCred - PRCredBuRCred + URCredRCredPr + URCredRCredVl + URCredHy


    ######################################## 
    # Pool of 3NOP in the rumen, Q3NOP, mol
    ######################################## 

    ### Production rates
    
    # Intake rate of 3NOP
    P3NOPfeed = Fr3NOP/1000.0f0 * feed/ 1000.0f0 * 0.00826
    
    ### Uptake rates
    # Uptake due to absorption, mol/d
    U3NOPAb = V3NOPAb * (vru ^ 0.75f0) / (1.0f0 + (M3NOPAb / C3NOP)^2.5 )

    # Uptake due to microbial degradation, mol/d
    vmapma = 0.0576f0
    U3NOPMi = (QMa + QMc ) * vmapma / (1.0f0 + m3nopma / C3NOP )

    # Uptake due to passage, mol/d
    U3NOPPa = Q3NOP * kFlPa
    
    # Uptake from methanogens, mol/d
    U3NOPMe = QMe * v3nopMCR * (pMCRactive) / (1.0f0 + ((m3nopMCRinact * (1 + CHy/jhy3nopMe)) /C3NOP)  )  

    
    ### Differential equation
    Q3NOP_dt = P3NOPfeed - U3NOPPa - U3NOPAb  - U3NOPMi - U3NOPMe




    ##################################### 
    #--Dissolved Hydrogen pool, QHy, mol  
    ##################################### 
    yhynadhnad = 1.0f0

    
    ### Production rates
    PHyRCredRCox = yhynadhnad * (URCredHy )

    
    ### Uptake rates
    UHyCH4 = vhych4 * QMe * pMCRactive / (1.0f0 +  (mhyhych4 / CHy ))
    UHyEv = QHy  * k_evap
    UHyPa = QHy * kFlPa
    UH2Ac = QHy * khyac / (1.0f0 + (mhyhyac / CHy) )

    ### Differential equation
    QHy_dt = PHyRCredRCox - UHyCH4 - UHyEv - UHyPa - UH2Ac 

    ############################### 
    #--Methanogens pool, QMe, gram 
    ############################### 

    yatphych4 = 0.125f0
    kPoPa = 0.5f0 * kSoPa
    ratpme = 0.1025f0  
    ratpmcr = 1.0f0

    ymeatp = YATP_0 + diff_atp / (1.0f0 +  (m_chyme_atp/CHy)^ o_chyme_atp )

    NSC = st_intake / (feed / 1000.0f0) + wr_intake / (feed / 1000.0f0)

    props_kmepa = minMeFl + (0.40f0 - minMeFl) / (1.0f0+((feed / 1000.0f0) / jdmiMeFl )^6.0f0)


    Me_Fl = props_kmepa / (1 + (CHy / jhyMePa)^2.0f0 )  
    Me_So = 0.20f0 + (1.0f0 - props_kmepa - 0.20f0) / 2.0f0 
    Me_Po = (1.0f0 - Me_So - Me_Fl)

    
    kmepa = Me_Po * (kSoPa * 0.5) + Me_Fl * kFlPa + Me_So * kSoPa
    # Missing 20% attached in rumen wall

            ########################## 
            # ATP-zero pool, QATP, mol 
            ########################## 
            ### Production rates
    PATPHyCH4 = UHyCH4 * yatphych4
    
            ### Uptake rates
    UATPMeMt = QMe * ratpme
    PMCRreactivation = QMCRinact * kMCRreact
    UATPMCRreact = PMCRreactivation * ratpmcr
    UATPMeGt = PATPHyCH4 - UATPMeMt - UATPMCRreact


    ### Production rates
    PMeHyCH4 = UATPMeGt * ymeatp
    
    ### Uptake rates
    UMeMePa = QMe * kmepa  

    
    ### Differential equation
    QMe_dt = PMeHyCH4 - UMeMePa
    



    ############################################### 
    # Proportion of MCR in methanogens, pMCRred, -
    ############################################### 
    cmcrme = 4.0f-7 # Concentration of MCR per gram of methanogen biomass


    #####################################
    # Pool of MCR active, QMCRactive, mol  
    #####################################
    
    ### Production rates
    PMCRactiveGrowth = PMeHyCH4 * cmcrme

    PMCRreactivation = QMCRinact * kMCRreact

    
    ### Uptake rates
    UMCRredMCRox = U3NOPMe / 2.0f0
    UMCRactivePa = UMeMePa * cmcrme * QMCRactive / (QMCRactive + QMCRinact)
    
    # Differential equation
    QMCRactive_dt = PMCRactiveGrowth + PMCRreactivation - UMCRredMCRox - UMCRactivePa
    

    #########################################
    #### Pool of MCR inactive, QMCRinact, mol 
    #########################################

    ### Production rates
    PMCRinact = UMCRredMCRox

    ### Uptake rates
    UMCRinactPa = UMeMePa * cmcrme * QMCRinact / (QMCRactive + QMCRinact)
    UMCRinactReact = PMCRreactivation

    # Differential equation
    QMCRinact_dt = PMCRinact - UMCRinactReact - UMCRinactPa

    #########################
    # Ethanol pool, Qeth, mol 
    #########################

    PEth = PAc * (2.0f0 - YCHAc)
    
    UEthabs = Qeth * kEthAb
    UEthPa = Qeth * kFlPa
    
    UEthBu = Qeth * kethbu 

    Qeth_dt = PEth - UEthabs - UEthPa - UEthBu

    

    return SVector{length(u)}(QRCred_dt, QRCox_dt, QHy_dt, QMe_dt, QAc_dt, QPr_dt, QBu_dt, QVl_dt, Q3NOP_dt, Qeth_dt, QMCRactive_dt, QMCRinact_dt)
end


# Function that contains the same model equations, but allows to extract auxiliary
# variables apart from pool quantities, such as daily rates (H2, CH4 production, etc.)
function model_eqs_3NOP_model_aux_vars( u, p, t)

    # Define the pools
    QRCred = u[1]       
    QRCox = u[2]
    QHy = u[3]
    QMe = u[4]
    QAc = u[5]
    QPr = u[6]
    QBu = u[7]
    QVl = u[8]
    Q3NOP = u[9]
    Qeth = u[10]
    QMCRactive = u[11]
    QMCRinact = u[12]

    pRCred = QRCred / (QRCred + QRCox)                      # Proportion of redox cofactors in reduced state
    pMCRactive =  QMCRactive / (QMCRactive + QMCRinact)     # Proportion of MCR enzyme in active state

    # MODEL INPUTS UNFOLDING
    feed, ndf_intake, st_intake, wr_intake, p_intake, la_intake, missfr_intake, pcell, 
    ndf_dig, st_dig, wr_dig, p_dig, pH,  FrAc,
    ffmCe, micr_n_fermented, UHaMMa, QMa, QMc, Fr3NOP = p


    ##################
    # Model parameters 
    ##################
    
    # Parameters VFA stoichiometry, for reference read Vivares et al (2026a) https://doi.org/10.3168/jds.2025-28000
    Y0AcW = 0.45
    Y0AcS = 0.54
    Y0AcH = 0.66
    Y0AcC = 0.73
    Y0AcP = 0.57

    Y0PrW = 0.12
    Y0PrS = 0.15
    Y0PrH = 0.08
    Y0PrC = 0.13
    Y0PrP = 0.26
    
    Y0BuW = 0.40
    Y0BuS = 0.21
    Y0BuH = 0.26
    Y0BuC = 0.13
    Y0BuP = 0.10
        
    Y0VlW = 0.03
    Y0VlS = 0.10
    Y0VlH = 0.00
    Y0VlC = 0.01
    Y0VlP = 0.07
    
    diffyacW = -0.20
    diffyprW = 0.04
    diffybuW = 0.12
    diffYVlW = 0.04
    mpRCredW = 0.38
    opRCredW = 6.9

    diffyacS = -0.17
    diffyprS = 0.06
    diffybuS = -0.01
    diffYVlS = 0.12
    mpRCredS = 0.32
    opRCredS = 7.0
    
    diffacW_ph = -0.22
    diffprW_ph = 0.32
    diffbuW_ph = - (diffacW_ph + diffprW_ph)
    mphacW = 5.68
    ophacW = 16.6
   
    diffacS_ph = -0.36
    diffprS_ph = 0.32
    diffbuS_ph = - (diffacS_ph + diffprS_ph)
    mphacS = 5.75
    ophacS = 27.3
    
    phem = 1.0f0 - pcell

    fS_NSC = (st_intake * st_dig) / (st_intake * st_dig + wr_intake * wr_dig) 
    fW_NSC = (wr_intake * wr_dig) / (st_intake * st_dig + wr_intake * wr_dig) 

    YRCredAc = 2.0f0
    YRCredBu = 2.0f0
    RRCredPr = 1.0f0
    RRCredVl = 2.0f0

    kRCOxH2 = 2.48f04
    vhych4 = 3.64
    mhyhych4 = 1.31f-5
    k_evap = 100.2

    jdmiMeFl = 30.4 
    minMeFl = 0.13
    
    # PARAMETERS RELATED TO 3-NITROOXYPROPANOL
    V3NOPAb = 6.4               # Max. fractional rate of 3NOP absorption
    M3NOPAb = 1.4f-4            # MichaelisMenten affinity of 3NOP  for 3NOP absorption
    m3nopma = 0.014 
    
    jhyMePa = 6.8f-4
    v3nopMCR = 0.011
    m3nopMCRinact = 3.7f-05  
    kMCRreact = 4.3
    jhy3nopMe = 2.8f-4
    
    mh2hexeth = 1.9f-03
    oh2hexeth = 1.9
    kethbu  = 9.1
    kEthAb = 16.8

    khyac = 858.0
    mhyhyac = 2.4f-03

    YATP_0 = 6.0 
    diff_atp = 0.52
    m_chyme_atp = 1.4f-03
    o_chyme_atp = 2.9

    vru = 47.86f0 + (1.759f0 * (feed/1000.0f0))     # L, Rumen volume
    kFlPa = 0.97f0 + (0.116f0 * (feed / 1000.0f0))  # /d, fluid passage rate
    kSoPa = 0.57f0 + (0.017f0 * (feed / 1000.0f0))  # /d, solid passage rate
    
    #######################################
    # Concentrations = Pool / Rumen volume
    #######################################
    CHy = QHy / vru
    CAc = QAc / vru
    CPr = QPr / vru
    CBu = QBu / vru
    CVl = QVl / vru
    C3NOP = Q3NOP / vru
    Ceth = Qeth / vru


    ###########################
    # Pool of acetate, QAc, mol
    ###########################
    ### Production rates
    YAcC = Y0AcC  
    YAcH = Y0AcH  
    YAcS = Y0AcS + diffyacS / (1.0f0 +  (mpRCredS / pRCred)^opRCredS)  + (diffacS_ph / (1.0f0 + (pH / mphacS)^ophacS)   )
    YAcW = Y0AcW + diffyacW / (1.0f0 +  (mpRCredW / pRCred)^opRCredW)  + (diffacW_ph / (1.0f0 +  (pH / mphacW)^ophacW)   )
    YAcP = Y0AcP 

    # Production of Acetate from cellulose fermentation, mol/d
    PAcC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe * YAcC

    # Production of Acetate from hemicellulose fermentation, mol/d
    PAcH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe * YAcH

    # Production of Acetate from Starch fermentation, mol/d
    PAcS = ((st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe *  YAcS ) + ( UHaMMa * 0.711 * YAcS * fS_NSC)

    # Production of Acetate from soluble carbohydrates fermentation, mol/d
    PAcW = (((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe *  YAcW ) + (UHaMMa * 0.711 * fW_NSC * YAcW)

    # Production of Acetate from protein fermentation, mol/d
    PAcP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented )* (YAcP)   

    

    # Total Acetate production from feed fermentation, mol/d
    YCHAc = 2.0f0 - 1.0f0 / (1.0f0 + (mh2hexeth / CHy) ^ oh2hexeth )
    PAc = (PAcC + PAcH + PAcS + PAcW) * YCHAc + (PAcP) * 1.07f0

    # Acetate production from reductive acetogenesis, mol/d
    UH2Ac = QHy * khyac / (1.0f0 + (mhyhyac / CHy) )
    PAcH2Ac = UH2Ac / 4.0f0

    # Acetate intake from feed (silage mostly)
    yacfac = 1.0f0 / 60.0f0
    Ac_intake = FrAc * feed / 1000.0f0 * yacfac

    ### Uptake rates
    
    # Uptake for absorption, mol/d  (Parameters derived from Dijkstra et al. 1992)      
    UAcCAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QAc / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UAcCPa = kFlPa * QAc
    # Uptake for butyrate together with ethanol, mol/d    
    UEthBu = Qeth * kethbu 
    UAcEthBu = UEthBu * 0.6
    

    # Differential equation

    QAc_dt = PAc + Ac_intake - UAcCAb - UAcCPa + PAcH2Ac - UAcEthBu

    ##############################
    # Pool of propionate, QPr, mol
    ##############################
    # Inputs
    YPrC = Y0PrC 
    YPrH = Y0PrH  
    YPrS = Y0PrS + diffyprS / (1.0f0 +  (mpRCredS / pRCred)^opRCredS)  + (diffprS_ph / (1.0f0 +  (pH / mphacS) ^ ophacS)  )
    YPrW = Y0PrW + diffyprW / (1.0f0 +  (mpRCredW / pRCred)^opRCredW) + diffprW_ph / (1.0f0 +  (pH / mphacW) ^ ophacW)
    YPrP = Y0PrP

    ### Production rates

    # Production of Propionate from cellulose fermentation, mol/d
    PPrC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe * YPrC  
    # Production of Propionate from hemicellulose fermentation, mol/d
    PPrH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe * YPrH  
    # Production of Propionate from Starch fermentation, mol/d
    PPrS = (st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe *  YPrS + ( UHaMMa * 0.711 * YPrS *fS_NSC)
    # Production of Propionate from soluble carbohydrates fermentation, mol/d
    PPrW = ((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe * YPrW  + (UHaMMa * 0.711 * fW_NSC * YPrW)
    # Production of Propionate from protein fermentation, mol/d
    PPrP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented )* ( YPrP)  
    # Total Propionate production from feed fermentation, mol/d
    PPr = (PPrC + PPrH + PPrS + PPrW) * 2.0f0 + (PPrP) * 1.07f0

    ### Uptake rates
    
    # Uptake for absorption, mol/d  (Parameters derived from Dijkstra et al. 1992)      
    UPrPAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QPr / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UPrPPa = kFlPa * QPr

    # Differential equation
    QPr_dt = PPr - UPrPAb - UPrPPa

    ############################ 
    # Pool of butyrate, QBu, mol
    ############################ 
    # Inputs
    YBuC = Y0BuC  
    YBuH = Y0BuH  
    YBuS = Y0BuS + diffybuS / (1.0f0 +  (mpRCredS / pRCred) ^ opRCredS) + (diffbuS_ph / (1.0f0 +  (pH / mphacS) ^ ophacS)  )
    YBuW =  Y0BuW + diffybuW / (1.0f0 +  (mpRCredW / pRCred) ^ opRCredW) + (diffbuW_ph / (1.0f0 +  (pH / mphacW) ^ ophacW)  )
    YBuP = Y0BuP

    ### Production rates

    # Production of Butyrate from cellulose fermentation, mol/d
    PBuC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe *  YBuC
    # Production of Butyrate from hemicellulose fermentation, mol/d
    PBuH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe *  YBuH
    # Production of Butyrate from Starch fermentation, mol/d
    PBuS = (st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe * YBuS + ( UHaMMa * 0.711 * YBuS *fS_NSC)
    # Production of Butyrate from soluble carbohydrates fermentation, mol/d
    PBuW = ((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe *  YBuW  + (UHaMMa * 0.711 * fW_NSC * YBuW)
    # Production of Butyrate from protein fermentation, mol/d
    PBuP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented )*   YBuP

    # Total Butyrate production from feed fermentation, mol/d
    PBu = PBuC + PBuH + PBuS + PBuW + PBuP * 0.535f0
    # Production of butyrate from ethanol and acetate, mol/d
    PBuEthAc = UEthBu * 0.8

    ### Uptake rates
    
    # Uptake for absorption, mol/d (Parameters derived from Dijkstra et al. 1992)   
    UBuBAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QBu / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UBuBPa = kFlPa * QBu

    # Differential equation
    QBu_dt = PBu - UBuBAb - UBuBPa + PBuEthAc

    ################################################
    # Pool of valerate and other minor VFA, QVl, mol
    ################################################ 
    # Inputs
    YVlC = Y0VlC  
    YVlH = Y0VlH  
    YVlS = Y0VlS + diffYVlS / (1.0f0 +  (mpRCredS / pRCred) ^ opRCredS)
    YVlW = Y0VlW + diffYVlW / (1.0f0 +  (mpRCredW / pRCred) ^ opRCredW)
    YVlP = Y0VlP  

    ### Production rates

    # Production of Valerate and other minor VFA from cellulose fermentation, mol/d
    PVlC = ndf_intake / 162.0f0 * pcell * ndf_dig * ffmCe *  YVlC
    # Production of Valerate and other minor VFA from hemicellulose fermentation, mol/d
    PVlH = ndf_intake / 162.0f0 * phem * ndf_dig * ffmCe *  YVlH
    # Production of Valerate and other minor VFA from Starch fermentation, mol/d
    PVlS = (st_intake / 162.0f0 * st_dig + la_intake / 90.08f0) * ffmCe *  YVlS + ( UHaMMa * 0.711 * YVlS * fS_NSC)
    # Production of Valerate and other minor VFA from soluble carbohydrates fermentation, mol/d
    PVlW = ((wr_intake / 162.0f0 * wr_dig) + (missfr_intake / 162.0f0 * (0.5f0 * ndf_dig + 0.5f0 * wr_dig))) * ffmCe * YVlW  + (UHaMMa * 0.711 * fW_NSC * YVlW)
    # Production of Valerate and other minor VFA from protein fermentation, mol/d
    PVlP = ((p_intake / 110.0f0 * p_dig  * ffmCe  )+ micr_n_fermented ) * YVlP

    # Total Valerate and other minor VFA production from feed fermentation, mol/d
    PVl = PVlC + PVlH + PVlS + PVlW + PVlP * 0.535f0

    ### Uptake rates
    
    # Uptake for absorption, mol/d  (Parameters derived from Dijkstra et al. 1992)     
    UVlVAb = 7.86 * (vru^0.75) / (1 + 0.338 / (QVl / vru)) / (1 + (pH / 6.45)^6.48)
    # Uptake for passage, mol/d    
    UVlVPa = kFlPa * QVl

    ### Differential equation
    QVl_dt = PVl - UVlVAb - UVlVPa


    ####################################################################### 
    # Pool of redox cofactors in the reduced state (i.e., NADH, Fdred), mol
    #######################################################################

    ### Production rates
    
    PRCredAcRCred = YRCredAc * PAc
    
    ### Uptake rates
    UEthBu = Qeth * kethbu 
    PBuEthAc = UEthBu * 0.8
    PRCredBuRCred = YRCredBu * PBu + PBuEthAc * 0.5

    # Uptake fluxes
    URCredRCredPr = RRCredPr * PPr
    URCredRCredVl = RRCredVl * PVl
    
    mhy_ft = 0.0005f0 + 0.0005f0 * pRCred
    Ft = 1.0f0 / (1.0f0 + ( CHy /mhy_ft)^3.0f0)
    URCredHy = QRCred * kRCOxH2 * Ft
    
    ### Differential equation
    QRCred_dt =  PRCredAcRCred + PRCredBuRCred - URCredRCredPr - URCredRCredVl - URCredHy

    #######################################################################
    # Pool of redox cofactors in the oxidized state (i.e., NAD, Fdox), mol
    #######################################################################
    ### Differential equation --> Inverse signs to QRCred_dt
    QRCox_dt = - PRCredAcRCred - PRCredBuRCred + URCredRCredPr + URCredRCredVl + URCredHy


    ######################################## 
    # Pool of 3NOP in the rumen, Q3NOP, mol
    ######################################## 

    ### Production rates
    
    # Intake rate of 3NOP
    P3NOPfeed = Fr3NOP/1000.0f0 * feed/ 1000.0f0 * 0.00826
    
    ### Uptake rates
    # Uptake due to absorption, mol/d
    U3NOPAb = V3NOPAb * (vru ^ 0.75f0) / (1.0f0 + (M3NOPAb / C3NOP)^2.5 )

    # Uptake due to microbial degradation, mol/d
    vmapma = 0.0576f0
    U3NOPMi = (QMa + QMc ) * vmapma / (1.0f0 + m3nopma / C3NOP )

    # Uptake due to passage, mol/d
    U3NOPPa = Q3NOP * kFlPa
    
    # Uptake from methanogens, mol/d
    U3NOPMe = QMe * v3nopMCR * (pMCRactive) / (1.0f0 + ((m3nopMCRinact * (1 + CHy/jhy3nopMe)) /C3NOP)  )  

    
    ### Differential equation
    Q3NOP_dt = P3NOPfeed - U3NOPPa - U3NOPAb  - U3NOPMi - U3NOPMe




    ##################################### 
    #--Dissolved Hydrogen pool, QHy, mol  
    ##################################### 
    yhynadhnad = 1.0f0

    
    ### Production rates
    PHyRCredRCox = yhynadhnad * (URCredHy )

    
    ### Uptake rates
    UHyCH4 = vhych4 * QMe * pMCRactive / (1.0f0 +  (mhyhych4 / CHy ))
    UHyEv = QHy  * k_evap
    UHyPa = QHy * kFlPa
    UH2Ac = QHy * khyac / (1.0f0 + (mhyhyac / CHy) )

    ### Differential equation
    QHy_dt = PHyRCredRCox - UHyCH4 - UHyEv - UHyPa - UH2Ac 

    ############################### 
    #--Methanogens pool, QMe, gram 
    ############################### 

    yatphych4 = 0.125f0
    kPoPa = 0.5f0 * kSoPa
    ratpme = 0.1025f0  
    ratpmcr = 1.0f0

    ymeatp = YATP_0 + diff_atp / (1.0f0 +  (m_chyme_atp/CHy)^ o_chyme_atp )

    NSC = st_intake / (feed / 1000.0f0) + wr_intake / (feed / 1000.0f0)

    props_kmepa = minMeFl + (0.40f0 - minMeFl) / (1.0f0+((feed / 1000.0f0) / jdmiMeFl )^6.0f0)


    Me_Fl = props_kmepa / (1 + (CHy / jhyMePa)^2.0f0 )  
    Me_So = 0.20f0 + (1.0f0 - props_kmepa - 0.20f0) / 2.0f0 
    Me_Po = (1.0f0 - Me_So - Me_Fl)

    
    kmepa = Me_Po * (kSoPa * 0.5) + Me_Fl * kFlPa + Me_So * kSoPa
    # Missing 20% attached in rumen wall

            ########################## 
            # ATP-zero pool, QATP, mol 
            ########################## 
            ### Production rates
    PATPHyCH4 = UHyCH4 * yatphych4
    
            ### Uptake rates
    UATPMeMt = QMe * ratpme
    PMCRreactivation = QMCRinact * kMCRreact
    UATPMCRreact = PMCRreactivation * ratpmcr
    UATPMeGt = PATPHyCH4 - UATPMeMt - UATPMCRreact


    ### Production rates
    PMeHyCH4 = UATPMeGt * ymeatp
    
    ### Uptake rates
    UMeMePa = QMe * kmepa  

    
    ### Differential equation
    QMe_dt = PMeHyCH4 - UMeMePa
    



    ############################################### 
    # Proportion of MCR in methanogens, pMCRred, -
    ############################################### 
    cmcrme = 4.0f-7 # Concentration of MCR per gram of methanogen biomass


    #####################################
    # Pool of MCR active, QMCRactive, mol  
    #####################################
    
    ### Production rates
    PMCRactiveGrowth = PMeHyCH4 * cmcrme

    PMCRreactivation = QMCRinact * kMCRreact

    
    ### Uptake rates
    UMCRredMCRox = U3NOPMe / 2.0f0
    UMCRactivePa = UMeMePa * cmcrme * QMCRactive / (QMCRactive + QMCRinact)
    
    # Differential equation
    QMCRactive_dt = PMCRactiveGrowth + PMCRreactivation - UMCRredMCRox - UMCRactivePa
    

    #########################################
    #### Pool of MCR inactive, QMCRinact, mol 
    #########################################

    ### Production rates
    PMCRinact = UMCRredMCRox

    ### Uptake rates
    UMCRinactPa = UMeMePa * cmcrme * QMCRinact / (QMCRactive + QMCRinact)
    UMCRinactReact = PMCRreactivation

    # Differential equation
    QMCRinact_dt = PMCRinact - UMCRinactReact - UMCRinactPa

    #########################
    # Ethanol pool, Qeth, mol 
    #########################

    PEth = PAc * (2.0f0 - YCHAc)
    
    UEthabs = Qeth * kEthAb
    UEthPa = Qeth * kFlPa
    
    UEthBu = Qeth * kethbu 

    Qeth_dt = PEth - UEthabs - UEthPa - UEthBu



    aux_vars_df = DataFrame()

    aux_vars_df.Fr3NOP = [Fr3NOP]

    aux_vars_df.ch4_rumen = [UHyCH4 / 4 * 16.043]
    aux_vars_df.h2_rumen = [UHyEv * 2.016]
    

    aux_vars_df.P3NOPfeed = [P3NOPfeed]
    aux_vars_df.U3NOPAb = [U3NOPAb]
    aux_vars_df.U3NOPMi = [U3NOPMi]
    aux_vars_df.U3NOPPa = [U3NOPPa]

    aux_vars_df.kmepa = [kmepa]
    aux_vars_df.ymeatp = [ymeatp]

    return aux_vars_df

end

############################################################## 
# Functions simulating the model and generating output dataset
##############################################################

# Initial states of pools
u0 = @SVector [0.001f0; 0.01f0; 0.00025f0; 30.0f0;  0.10f0;  0.10f0;  0.10f0;  0.10f0; 0.00000000001f0; 0.00001f0; 5.04f-4; 0.0000f-6  ] 

# Time span to simulate the model, up to 50 days
tspan = (0.0f0, 60.0f0)  # Time span

function add_aux_vars(df_Qs::DataFrame, df_inputs::DataFrame)
    #results = Vector{Any}(undef, nrow(df))
    results_df = DataFrame()
    for i in 1:nrow(df_Qs)        
        row_Qs = df_Qs[i, ["QRCred", "QRCox", "QHy", "QMe", "QAc", "QPr", "QBu", "QVl", "Q3NOP", "QEth", "QMCRactive", "QMCRinact"]]
        row_inputs = Vector(df_inputs[i,:])

        aux_vars = model_eqs_3NOP_model_aux_vars(Vector(row_Qs), row_inputs, 49)
        res_row = hcat(DataFrame(df_Qs[i, :]), aux_vars)
        push!(results_df, res_row[1,:])
        
    end
    return results_df
end    

# Function to simulate the differential equations of the model
function simulation_3NOP_model(inputs_df)
        # Solve ensemble of experiments.
    p = SVector{length(Float32.(Vector(inputs_df[1, :])))}(Float32.(Vector(inputs_df[1, :])))
    prob = ODEProblem{false}(model_eqs_3NOP_model, u0, tspan, p)
    
    prob_func = (prob, i, repeat) -> remake(prob, p = SVector{ length(Float32.(Vector(inputs_df[1, :]))) }(Float32.(Vector(inputs_df[i, :]))) )
    monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy = false)
    sol = solve(monteprob, GPURosenbrock23(), EnsembleGPUKernel(CUDA.CUDABackend()),
        trajectories = nrow(inputs_df), dt = 0.00000001f0,  maxiters = 300.0e6,
        saveat = 1.0f0)
 

        
    # Export observable variables from simulation.
    QRCred = [sol_i.u[1][1] for sol_i in sol]
    QRCox = [sol_i.u[end][2] for sol_i in sol]
    QHy = [sol_i.u[end][3] for sol_i in sol]
    QMe = [sol_i.u[end][4] for sol_i in sol]
    QAc = [sol_i.u[end][5] for sol_i in sol]
    QPr = [sol_i.u[end][6] for sol_i in sol]
    QBu = [sol_i.u[end][7] for sol_i in sol]
    QVl = [sol_i.u[end][8] for sol_i in sol]
    Q3NOP = [sol_i.u[end][9] for sol_i in sol]
    QEth = [sol_i.u[end][10] for sol_i in sol]
    QMCRactive = [sol_i.u[end][11] for sol_i in sol]
    QMCRinact = [sol_i.u[end][12] for sol_i in sol]
    pMCRred = QMCRactive ./ (QMCRactive .+ QMCRinact)

    rAc = QAc ./ (QAc .+ QPr .+ QBu .+ QVl) .* 100
    rPr = QPr ./ (QAc .+ QPr .+ QBu .+ QVl) .* 100
    rBu = QBu ./ (QAc .+ QPr .+ QBu .+ QVl) .* 100
    rVl = QVl ./ (QAc .+ QPr .+ QBu .+ QVl) .* 100

    output_df = DataFrame()
    output_df[!, :QRCred] = QRCred 
    output_df[!, :QRCox] = QNAD 
    output_df[!, :QHy] = QHy
    output_df[!, :QMe] = QMe 
    output_df[!, :QAc] = QAc 
    output_df[!, :QPr] = QPr 
    output_df[!, :QBu] = QBu 
    output_df[!, :QVl] = QVl 
    output_df[!, :QVFA] = QAc .+ QPr .+ QBu .+ QVl 
    output_df[!, :pred_ac] = rAc
    output_df[!, :pred_pr] = rPr
    output_df[!, :pred_bu] = rBu
    output_df[!, :pred_vl] = rVl

    output_df[!, :Q3NOP] = Q3NOP 
    output_df[!, :QEth] = QEth 
    output_df[!, :QMCRactive] = QMCRactive 
    output_df[!, :QMCRinact] = QMCRinact 
    output_df[!, :pMCRred] = pMCRred


    output_df = add_aux_vars(output_df, inputs_df)
    return output_df
end


output_df = simulation_3NOP_model(inputs_df)

################################ 
# Visualization of model outputs
################################
default(color = :black, label = false)
plt = plot(layout = (2,2), size = (600, 600), grid = false, label = false, xlabel = "3-NOP dose (mg/kg)")


scatter!(plt[1], output_df.Fr3NOP, output_df.ch4_rumen, ylabel = "rumen CH4 production, g/d")
plot!(plt[1], output_df.Fr3NOP, output_df.ch4_rumen)

scatter!(plt[2], output_df.Fr3NOP, output_df.h2_rumen, ylabel = "rumen H2 production, g/d")
plot!(plt[2], output_df.Fr3NOP, output_df.h2_rumen)

scatter!(plt[3], output_df.Fr3NOP, output_df.pMCRred, ylabel = "Proportion MCR active, -")
plot!(plt[3], output_df.Fr3NOP, output_df.pMCRred)

scatter!(plt[4], output_df.Fr3NOP, output_df.pred_ac, ylabel = "Proportion acetate, -")
plot!(plt[4], output_df.Fr3NOP, output_df.pred_ac)




