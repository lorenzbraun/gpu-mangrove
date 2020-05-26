# Makefile for training the models for time
#
# Set this command if you want to use a scheduler (like e.g. slurm) to 
# distribute the task of data processing and training
# CMDP=srun -o slurm-%j.out
#
# GPUs
GPUS=K20 TitanXp P100 V100 GTX1650
# Threshold of how many samples are being used for a bench,app,dataset tuple
THRESHOLD=100
# Aggregation method for combining multiple measurements into one sample
# choose mean or median
AGG=median
# Folder where data is stored
DDIR=data

.PHONY: 
all: time_models power_models

.PHONY: 
time_models: $(foreach GPU,$(GPUS),$(DDIR)/time_model_$(GPU)_$(AGG)_$(THRESHOLD).pkl) 
.PHONY: 
power_models: $(foreach GPU,$(GPUS),$(DDIR)/power_model_$(GPU)_$(AGG)_$(THRESHOLD).pkl) 

.PHONY:
loo: time_loo power_loo
.PHONY:
time_loo: $(foreach GPU,$(GPUS),$(DDIR)/time_loo_$(GPU)_$(AGG)_$(THRESHOLD).pkl)
.PHONY:
power_loo: $(foreach GPU,$(GPUS),$(DDIR)/power_loo_$(GPU)_$(AGG)_$(THRESHOLD).pkl)

# Time Model Targets
define create_samples_t
$(DDIR)/time_samples_$1_$2.db: $(DDIR)/FeatureTable-0.3.db $(DDIR)/KernelTime-$1.db
	$(CMDP) python mangrove.py process --fdb $(DDIR)/FeatureTable-0.3.db --mdb $(DDIR)/KernelTime-$1.db -o $$@ -g lconf -a median
endef

define create_samplesf_t
$(DDIR)/time_samplesf_$1_$2_$3.pkl: $(DDIR)/time_samples_$1_$2.db
	$(CMDP) python mangrove.py filter -i $$^ -o $$@ -t $3
endef

define create_models_t
$(DDIR)/time_model_$1_$2_$3.pkl: $(DDIR)/time_samplesf_$1_$2_$3.pkl
	$(CMDP) python mangrove.py cv -i $$^ -o $$@ -r $(DDIR)/time_cv-res_$1_$2_$3.pkl -t 30 -s 3 -k 5
endef

define create_loo_t
$(DDIR)/time_loo_$1_$2_$3.pkl: $(DDIR)/time_model_$1_$2_$3.pkl $(DDIR)/time_samplesf_$1_$2_$3.pkl
	$(CMDP) python mangrove.py loo -i $(DDIR)/time_samplesf_$1_$2_$3.pkl -m $(DDIR)/time_model_$1_$2_$3.pkl -o $$@
endef

$(foreach GPU,$(GPUS),$(eval $(call create_samples_t,$(GPU),$(AGG))))
$(foreach GPU,$(GPUS),$(eval $(call create_samplesf_t,$(GPU),$(AGG),$(THRESHOLD))))
$(foreach GPU,$(GPUS),$(eval $(call create_models_t,$(GPU),$(AGG),$(THRESHOLD))))
$(foreach GPU,$(GPUS),$(eval $(call create_loo_t,$(GPU),$(AGG),$(THRESHOLD))))

# Power Model Targets
define create_samples_p
$(DDIR)/power_samples_$1_$2.db: $(DDIR)/FeatureTable-0.3.db $(DDIR)/KernelPower-$1.db
	$(CMDP) python mangrove.py process --fdb $(DDIR)/FeatureTable-0.3.db --mdb $(DDIR)/KernelPower-$1.db -o $$@ -g lconf -a median
endef

define create_samplesf_p
$(DDIR)/power_samplesf_$1_$2_$3.pkl: $(DDIR)/power_samples_$1_$2.db
	$(CMDP) python mangrove.py filter -i $$^ -o $$@ -t $3
endef

define create_models_p
$(DDIR)/power_model_$1_$2_$3.pkl: $(DDIR)/power_samplesf_$1_$2_$3.pkl
	$(CMDP) python mangrove.py cv -i $$^ -o $$@ -r $(DDIR)/power_cv-res_$1_$2_$3.pkl -t 30 -s 3 -k 5
endef

define create_loo_p
$(DDIR)/power_loo_$1_$2_$3.pkl: $(DDIR)/power_model_$1_$2_$3.pkl $(DDIR)/power_samplesf_$1_$2_$3.pkl
	$(CMDP) python mangrove.py loo -i $(DDIR)/power_samplesf_$1_$2_$3.pkl -m $(DDIR)/power_model_$1_$2_$3.pkl -o $$@
endef

$(foreach GPU,$(GPUS),$(eval $(call create_samples_p,$(GPU),$(AGG))))
$(foreach GPU,$(GPUS),$(eval $(call create_samplesf_p,$(GPU),$(AGG),$(THRESHOLD))))
$(foreach GPU,$(GPUS),$(eval $(call create_models_p,$(GPU),$(AGG),$(THRESHOLD))))
$(foreach GPU,$(GPUS),$(eval $(call create_loo_p,$(GPU),$(AGG),$(THRESHOLD))))
