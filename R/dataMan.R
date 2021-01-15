# Household mixing matrix
cont <- read.csv("United_States_subnational_New_York_F_household_setting_85.csv", header=F)

cont <- cbind(paste0(0:84, " to ", 0:84), cont)
colnames(cont) <- c("Age group", paste0(0:84, " to ", 0:84))
write.csv(cont,"contMat.csv", row.names = F)


#Househols size distribution
hh <- data.frame(size=1:7, number=c(2270961,
                                    2337180,
                                    1192593,
                                    929445,
                                    419287,
                                    170255,
                                    127091))
write.csv(hh, file="households.csv", row.names = F)

##### Source: US Cencus: ACS 2019 1-year Table B11016
##### Citation: U.S. Census Bureau (2019). Household Type by Household Size American Community Survey 1-year estimates. Retrieved from <https://censusreporter.org>


##### Death data NY
deaths <- read.csv("raw/us-states_20200106.csv")
deathsNY <- subset(deaths, state=="New York", -c(state, cases, fips))
colnames(deathsNY) <- c("date", "cum_deaths")

deathsNY <- mutate(deathsNY, new_deaths = c(0, diff(cum_deaths)))

deathsNYFW <- subset(deathsNY, date<as.Date("2020-07-20"))
deathsNYSW <- subset(deathsNY, date<=as.Date("2020-07-20"))

write.csv(deathsNY, "deathsNY20200106.csv")
write.csv(deathsNYFW, "deathsNYFW20200106.csv")
write.csv(deathsNYSW, "deathsNYSW20200106.csv")

###### Householder ages
###### U.S. Census Bureau (2019). Age of Householder by Household Income in the Past 12 Months (In 2019 Inflation-adjusted Dollars) American Community Survey 1-year estimates. Retrieved from <https://censusreporter.org>
hhage <- data.frame(age=0:84, number= c(0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        28176,
                                        28176,
                                        28176,
                                        28176,
                                        28176,
                                        28176,
                                        28176,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        116636,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        142676,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167,
                                        103167))

write.csv(hhage, "ageref.csv", row.names = F)

















