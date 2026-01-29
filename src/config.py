# config.py

# ============================================================================= 
# CORE VARIABLE GROUPS 
# =============================================================================

correlation_cols = ['diagnosis', 'cdrglob', 'naccgds']

demo_cols = ['age', 'educ', 'female']

severity_cols = [
    'delsev', 'hallsev', 'agitsev', 'depdsev', 'anxsev', 'elatsev',
    'apasev', 'disnsev', 'irrsev', 'motsev', 'nitesev', 'appsev'
]

activity_cols = ['bills', 'taxes', 'shopping', 'games', 'stove', 
                     'mealprep', 'events', 'payattn', 'remdates', 'travel']

# ============================================================================= 
# BRAIN VOLUME GROUPINGS 
# =============================================================================

volume_groups = {
    "CSF Volumes": ['naccicv', 'csfvol'],
    "Cortex Volumes": ['frcort', 'lparcort', 'rparcort', 'ltempcor', 'rtempcor', 'lcac', 'rcac'],
    "Lobes & Regions": ['lhippo', 'rhippo', 'lent', 'rent', 'lparhip', 'rparhip', 'lposcin', 'rposcin']
}

brain_cols = [
    'csfvol', 'frcort', 'lparcort', 'rparcort', 'ltempcor', 'rtempcor',
    'rcac', 'lhippo', 'rhippo', 'lent', 'rent',
    'lparhip', 'rparhip', 'lposcin', 'rposcin'
]

# ============================================================================= 
# STATISTICAL PARAMETERS 
# =============================================================================

chi_square_alpha = 0.05

# ============================================================================= 
# DEFAULT COVARIATES 
# ============================================================================= 

covariates = ['female', 'educ']
