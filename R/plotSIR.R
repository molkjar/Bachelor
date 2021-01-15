#Load required libraries
library(ggplot2)
library(deSolve)
library(reshape2)
library(extrafont)
font_import()

theme_set(theme_bw())

# Model inputs

init=c(S=999999,I=1,R=0)
pars=c(gamma=0.30,beta=0.7)

# Time points

time=1:100

sir <- function(time,state,parameters){
  with(as.list(c(state,parameters)),{
    N=S+I+R
    dS=-beta*I*S/N
    dI=beta*I*S/N-gamma*I
    dR=gamma*I
    
    return(list(c(dS,dI,dR)))
  }
  )
}

# Solve
output<-as.data.frame(ode(y=init,func = sir_model,parms=pars,times = time))


out_long=melt(output,id="time")


# Plot

ggplot(data = out_long, aes(x = time, y = value/1000000, colour = variable, group = variable)) +  
  geom_line(size=1) +
  xlab("Time (days)")+ylab("Proportion of the population")+
  scale_color_discrete(name="Compartment") +
  theme(text=element_text(size=14,  family="Libre Caslon Text"))

ggsave("SIR_curves.png")

