# DOE Reference Building Plotting
#
# Written by: Jay H Arehart
# Written on: Nov 13, 2019
#
#
# Script Description:
#   Creating figures for the DOE Reference Building LCA study.
#
#
# Output
#   Publication ready figures

# Load libraries
library(ggplot2)
library(tidyverse)
library(reshape2)
library(dplyr)
library(tidyr)
library(rio)
library(RColorBrewer)
library(data.table)
library(scales)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


setwd("~/PycharmProjects/DOE_Ref_Materials/Results Summary")

# Import datasets from excel into dataframes
data_list <- import_list("Summary by Typology.xlsx")
summary_df<- data_list$summary
operational_df <- data_list$site_energy_GJ
operational_norm_df <- data_list$site_energy_MJ_m2
by_surface_df <- data_list$summary_by_surface
by_material_df <- data_list$summary_by_material
grid_factor_df <- data_list$grid_factor
rm(data_list)

# Contributionary Plots ----
#   Plots for contribution of different elements (EE/EC) to total EE/EC

# Specify buildings to be plotted
buildings_to_plot <- c('Large Office','Medium Office','Small Office','High-Rise Apartment','Mid-Rise Apartment')
czs_to_plot <- c('2A_Tampa', '4B_Albuquerque', '4C_Seattle', '6A_Rochester', '7_InternationalFalls')

# Plot subset of buildings for EE by material
by_material_df_ordered = by_material_df[order(by_material_df$EE_normalized, decreasing=T),]
by_material_df_ordered$new_category <- factor(by_material_df_ordered$category, levels=rev(c('Concrete','Window','Gypsum','Carpet','Metal','Insulation','Wood','Ceiling Tile','Stucco','Asphalt','Other')), ordered=T)
by_material_df_ordered$Building <- factor(by_material_df_ordered$Building, levels=c('Large Office','Medium Office','Small Office','High-Rise Apartment','Mid-Rise Apartment','Large Hotel','Small Hotel','Hospital','OutPatient HealthCare','Stand-alone Retail','Strip Mall','Quick Service Restaurant','Full Service Restaurant','Primary School','Secondary School','Warehouse'), ordered=T)
by_material_df_ordered_new <- subset(by_material_df_ordered, Building %in% buildings_to_plot)

EE_by_mat <- ggplot(by_material_df_ordered_new, aes(x = Building, y = EE_normalized, fill=new_category)) +
  geom_bar(position='stack', stat='identity', color = 'black') +
  geom_bar(stat='identity', aes(fill=new_category), color ='transparent') +
  scale_fill_brewer(palette="Set3", direction=-1, name = "Material") +
  theme_bw(base_size = 10) +
  scale_y_continuous(name=expression("Embodied Energy (MJ/m"^2*" )" ), breaks=c(0,200,400,600,800,1000,1200,1400,1600)) +
  scale_x_discrete(labels = wrap_format(8)) +
  theme(axis.text.x = element_text(angle=90,hjust=1, color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(legend.position='none') +
  ggtitle('(a)')

ggsave('Figures/EE_by_mat.png', plot = EE_by_mat, dpi = 440, width=3, height=3.5, units = "in")

## Plot subset of buildings for EC by material
by_material_df_ordered = by_material_df[order(by_material_df$EC_normalized, decreasing=T),]
by_material_df_ordered$new_category <- factor(by_material_df_ordered$category, levels=rev(c('Concrete','Window','Gypsum','Carpet','Metal','Insulation','Wood','Ceiling Tile','Stucco','Asphalt','Other')), ordered=T)
by_material_df_ordered$Building <- factor(by_material_df_ordered$Building, levels=c('Large Office','Medium Office','Small Office','High-Rise Apartment','Mid-Rise Apartment','Large Hotel','Small Hotel','Hospital','OutPatient HealthCare','Stand-alone Retail','Strip Mall','Quick Service Restaurant','Full Service Restaurant','Primary School','Secondary School','Warehouse'), ordered=T)
# Specify buildings to be plotted
by_material_df_ordered_new <- subset(by_material_df_ordered, Building %in% buildings_to_plot)


EC_by_mat <- ggplot(by_material_df_ordered_new, aes(x = Building, y = EC_normalized, fill=new_category)) +
  geom_bar(position='stack', stat='identity', color = 'black') +
  geom_bar(stat='identity', aes(fill=new_category), color ='transparent') +
  scale_fill_brewer(palette="Set3", direction=-1, name = "Material") +
  ggtitle('(b)') +
  theme_bw(base_size = 10) +
  scale_y_continuous(name=expression("Embodied Carbon (kgCO"[2]*"e/m"^2*")" ), breaks=c(0,20,40,60,80,100,120,140,160)) +
  scale_x_discrete(labels = wrap_format(8)) +
  theme(axis.text.x = element_text(angle=90, hjust=1, color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  guides(fill = guide_legend(ncol=1))
ggsave('Figures/EC_by_mat.png', plot = EC_by_mat, dpi = 440, width=4, height=3.25, units = "in")


ggsave('Figures/by_mat.png', plot = multiplot(EE_by_mat, EC_by_mat, cols=2), dpi = 440, width=7, height=3.5, units = "in")

 ## Plot subset buildings for EE by surface
by_surface_df_ordered = by_surface_df[order(by_surface_df$EE_normalized, decreasing=T),]
by_surface_df_ordered$new_category <- factor(by_surface_df_ordered$Surface_Type, levels=rev(c('Floor','Windows','Ceiling','Wall','Roof')), ordered=T)
by_surface_df_ordered$Building <- factor(by_surface_df_ordered$Building, levels=c('Large Office','Medium Office','Small Office','High-Rise Apartment','Mid-Rise Apartment','Large Hotel','Small Hotel','Hospital','OutPatient HealthCare','Stand-alone Retail','Strip Mall','Quick Service Restaurant','Full Service Restaurant','Primary School','Secondary School','Warehouse'), ordered=T)
by_surface_df_ordered_new <- subset(by_surface_df_ordered, Building %in% buildings_to_plot)

EE_by_surf <- ggplot(by_surface_df_ordered_new, aes(x = Building, y = EE_normalized, fill=new_category)) +
  geom_bar(position='stack', stat='identity', color = 'black') +
  geom_bar(stat='identity', aes(fill=new_category), color ='transparent') +
  scale_fill_brewer(palette="Accent", direction=-1, name = "Surface Type") +
  theme_bw(base_size = 10) +
  scale_y_continuous(name=expression("Embodied Energy (MJ/m"^2*" )" ), breaks=c(0,200,400,600,800,1000,1200,1400,1600)) +
  scale_x_discrete(labels = wrap_format(8)) +
  theme(axis.text.x = element_text(angle=90,hjust=1, color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(legend.position='none') +
  ggtitle('(a)')

ggsave('Figures/EE_by_surf.png', plot = EE_by_surf, dpi = 440, width=2.75, height=3, units = "in")


## Plot subset of buildings for EC by surface
by_surface_df_ordered = by_surface_df[order(by_surface_df$EC_normalized, decreasing=T),]
by_surface_df_ordered$new_category <- factor(by_surface_df_ordered$Surface_Type, levels=rev(c('Floor','Windows','Ceiling','Wall','Roof')), ordered=T)
by_surface_df_ordered$Building <- factor(by_surface_df_ordered$Building, levels=c('Large Office','Medium Office','Small Office','High-Rise Apartment','Mid-Rise Apartment','Large Hotel','Small Hotel','Hospital','OutPatient HealthCare','Stand-alone Retail','Strip Mall','Quick Service Restaurant','Full Service Restaurant','Primary School','Secondary School','Warehouse'), ordered=T)
by_surface_df_ordered_new <- subset(by_surface_df_ordered, Building %in% buildings_to_plot)

EC_by_surf <- ggplot(by_surface_df_ordered_new, aes(x = Building, y = EC_normalized, fill=new_category)) +
  geom_bar(position='stack', stat='identity', color = 'black') +
  geom_bar(stat='identity', aes(fill=new_category), color ='transparent') +
  scale_fill_brewer(palette="Accent", direction=-1, name = "Surface Type") +
  theme_bw(base_size = 10) +
  scale_y_continuous(name=expression("Embodied Carbon (kgCO"[2]*"e/m"^2*")" ), breaks=c(0,20,40,60,80,100,120,140,160)) +
  scale_x_discrete(labels = wrap_format(8)) +
  theme(axis.text.x = element_text(angle=90,hjust=1, color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  ggtitle('(b)')

ggsave('Figures/EC_by_surf.png', plot = EC_by_surf, dpi = 440, width=4, height=3.5, units = "in")

ggsave('Figures/by_surf.png', plot = multiplot(EE_by_surf, EC_by_surf, cols=2), dpi = 440, width=7, height=3.5, units = "in")



# Operational vs Embodied Plots ----

# Getting summary of embodied data
EE_norm_subset_df = subset(summary_df, Building %in% buildings_to_plot)
EE_norm_subset_df$Building <- factor(EE_norm_subset_df$Building, levels=c('Large Office','Medium Office','Small Office','High-Rise Apartment','Mid-Rise Apartment','Large Hotel','Small Hotel','Hospital','OutPatient HealthCare','Stand-alone Retail','Strip Mall','Quick Service Restaurant','Full Service Restaurant','Primary School','Secondary School','Warehouse'), ordered=T)
EE_norm_subset_df = EE_norm_subset_df[,c(1,5)]

EC_norm_subset_df = subset(summary_df, Building %in% buildings_to_plot)
EC_norm_subset_df$Building <- factor(EC_norm_subset_df$Building, levels=c('Large Office','Medium Office','Small Office','High-Rise Apartment','Mid-Rise Apartment','Large Hotel','Small Hotel','Hospital','OutPatient HealthCare','Stand-alone Retail','Strip Mall','Quick Service Restaurant','Full Service Restaurant','Primary School','Secondary School','Warehouse'), ordered=T)
EC_norm_subset_df = EC_norm_subset_df[,c(1,6)]


# Number of years for operational analyis
years = 1

operational_norm_subset_df = subset(operational_norm_df, Building %in% buildings_to_plot)
operational_norm_subset_df = select(operational_norm_subset_df,'Building', czs_to_plot)

OE_total_df = operational_norm_subset_df[, 2:ncol(operational_norm_subset_df)]*years     # Total energy consumed over lifetime [MJ/m2]
grid_factor_simple = grid_factor_df[,c(1,4)]  # kg CO2/MJ
grid_factor_simple = subset(grid_factor_simple, Location %in% czs_to_plot)
OC_total_df = data.frame(mapply(`*`,OE_total_df, grid_factor_simple$`eGRID 2016 (kg CO2/MJ)`))  # [kg CO2/m2]

  # Add back in Building names
OE_total_df = cbind(operational_norm_subset_df$Building, OE_total_df)
colnames(OE_total_df)[1] = 'Building'
OC_total_df = cbind(operational_norm_subset_df$Building, OC_total_df)
colnames(OC_total_df)[1] = 'Building'
colnames(OC_total_df) = colnames(OE_total_df)

# Creating total energy dataframe
Total_energy_df = cbind(EE_norm_subset_df,OE_total_df[, 2:ncol(operational_norm_subset_df)])  # MJ/m2
Total_energy_df_ordered = Total_energy_df[order(Total_energy_df$`2A_Tampa`, decreasing=T),]
Total_energy_df_melt = melt(Total_energy_df_ordered, id='Building', value.name='Energy_MJ')
Total_energy_df_melt$variable <- factor(Total_energy_df_melt$variable, levels = c('EE_normalized','7_InternationalFalls','6A_Rochester','4B_Albuquerque','2A_Tampa','4C_Seattle'))


Total_carbon_df = cbind(EC_norm_subset_df,OC_total_df[, 2:ncol(OC_total_df)])  # kg CO2/m2
Total_carbon_df_ordered = Total_carbon_df[order(Total_carbon_df$`2A_Tampa`, decreasing=T),]
Total_carbon_df_melt = melt(Total_carbon_df_ordered, id='Building', value.name='Carbon')
Total_carbon_df_melt$variable <- factor(Total_carbon_df_melt$variable, levels = c('EC_normalized','4B_Albuquerque','7_InternationalFalls','2A_Tampa','6A_Rochester','4C_Seattle'))


# Plotting operational vs. embodied energy/carbon

my_cols = brewer.pal(1+length(czs_to_plot), name='Set2')
my_cols_ee = c("grey40", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F")
my_cols_ec = c("grey40", "#E78AC3", "#FC8D62", "#A6D854", "#8DA0CB", "#FFD92F")

TE <- ggplot(Total_energy_df_melt, aes(x = Building, y = Energy_MJ, fill=variable )) +
  geom_bar(position='dodge', stat='identity', color = 'black') +
  scale_fill_manual(values = my_cols_ee, name = "EE and OE by Location") +
  theme_bw(base_size = 16) +
  scale_y_continuous(name=expression("Energy Consumption (MJ/m"^2*" )" )) +
  theme(axis.text.x = element_text(angle=90,hjust=1, color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(legend.position = c(0.25, 0.75),
        legend.background = element_blank(),
        legend.box.background = element_rect(colour = "black"))
ggsave('Figures/EE_vs_OE_1year.png', plot = TE, dpi = 440, width=7, height=7, units = "in")
plot(TE)

TC <- ggplot(Total_carbon_df_melt, aes(x = Building, y = Carbon, fill=variable )) +
  geom_bar(position='dodge', stat='identity', color = 'black') +
  scale_fill_manual(values = my_cols_ec, name = "EC and OC by Location") +
  theme_bw(base_size = 16) +
  scale_y_continuous(name=expression("GWP (kgCO"[2]*"e/m"^2*")" )) +
  theme(axis.text.x = element_text(angle=90,hjust=1, color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(legend.position = c(0.25, 0.75),
        legend.background = element_blank(),
        legend.box.background = element_rect(colour = "black"))
ggsave('Figures/EC_vs_OC_1year.png', plot = TC, dpi = 440, width=7, height=7, units = "in")
plot(TC)


# ---- Box plots of total vs CLF bencmarking study:

CLF_df <- data.frame(x=c("CLF - Office","CLF - Multi-Family"), min=c(10,42), low=c(266,342), mid=c(396,453), top=c(515,665), max=c(783,974))


benchmark_office <- ggplot() +
  geom_boxplot(data=CLF_df[1,],
               aes(x=x, ymin=min, lower=low, middle=mid, upper=top, ymax=max),
               stat = "identity",
               fill='gray70') +
  geom_bar(data=EC_norm_subset_df[grep("Office", EC_norm_subset_df$Building),],
           aes(x=Building, y=EC_normalized),
           stat = "identity",
           fill=c('#a6cee3','#1f78b4','#b2df8a'),
           color='black') +
  theme_bw(base_size = 10) +
  scale_y_continuous(name=expression("Embodied Carbon (kgCO"[2]*"e/m"^2*")" ), limit=c(0,1000)) +
  scale_x_discrete(labels = wrap_format(8)) +
  theme(axis.text.x = element_text(color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  ggtitle('(a)')
plot(benchmark_office)

ggsave('Figures/Benchmark_office.png', plot = benchmark_office, dpi = 440, width=4, height=4, units = "in")

benchmark_apartment <- ggplot() +
  geom_boxplot(data=CLF_df[2,],
               aes(x=x, ymin=min, lower=low, middle=mid, upper=top, ymax=max),
               stat = "identity",
               fill='gray70') +
  geom_bar(data=EC_norm_subset_df[grep("Apar", EC_norm_subset_df$Building),],
           aes(x=Building, y=EC_normalized),
           stat = "identity",
           fill=c('#fb9a99','#33a02c'),
           color='black') +
  theme_bw(base_size = 10) +
  scale_y_continuous(name=expression("Embodied Carbon (kgCO"[2]*"e/m"^2*")" ), limit=c(0,1000)) +
  scale_x_discrete(labels = wrap_format(8)) +
  theme(axis.text.x = element_text(color = 'black'),
        axis.title.x = element_blank(),
        axis.text.y = element_text(color = 'black')) +
  # theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  ggtitle('(b)')
plot(benchmark_apartment)

multiplot(benchmark_office, benchmark_apartment, cols=2)
ggsave('Figures/Benchmarks.png', plot = multiplot(benchmark_office, benchmark_apartment, cols=2), dpi = 440, width=7, height=4, units = "in")
 
