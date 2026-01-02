import re

#Get brand of cpu model by word matching
def getModelBrandCPU(model):
    model = model.replace("-"," ").lower().strip()
    model = model.split()
    intelWords = ["intel","ıntel","i3","i5","i7","i9","core"]
    amdWords = ["amd","ryzen", "pro", "threadripper"]
    appleWords = ["apple", "pro", "max", "ultra", "m1", "m2", "m3", "m4", "m5"]
    wordCount = {"Intel":0, "AMD":0, "Apple":0}
    for word in model:
        if word in intelWords:
            wordCount["Intel"] += 1
        if word in amdWords:
            wordCount["AMD"] += 1
        if word in appleWords:
            wordCount["Apple"] += 1
    maxValue = max(wordCount.values())
    if maxValue > 0:
        return list(wordCount.keys())[list(wordCount.values()).index(maxValue)]
    else:
        return None

#Get brand of gpu model by word matching
def getModelBrandGPU(model):
    model = model.replace("-"," ").lower().strip()
    model = model.split()
    nvidiaWords = ["rtx","gtx","geforce","ti","super","titan"]
    amdWords = ["rx","radeon","vega","xt"]
    intelWords = ["arc"]
    appleWords = ["apple"]
    wordCount = {"NVIDIA":0, "AMD":0, "Intel":0, "Apple":0}
    for word in model:
        if word in nvidiaWords:
            wordCount["NVIDIA"] += 1
        if word in amdWords:
            wordCount["AMD"] += 1
        if word in intelWords:
            wordCount["Intel"] += 1
        if word in appleWords:
            wordCount["Apple"] += 1
    maxValue = max(wordCount.values())
    if maxValue > 0:
        return list(wordCount.keys())[list(wordCount.values()).index(maxValue)]
    else:
        return None

#Extracts cpu generation from model
def extract_cpu_gen(model):
    brand = getModelBrandCPU(model)
    model = str(model)

    #If the brand is Apple, extract the number after the letter M
    #Example: Apple M3 Pro, gen = 3
    if brand == "Apple":
        match = re.search(r"[Mm](\d)", model)
        return int(match.group(1)) if match else None

    #Otherwise, extract the digits except last 3.
    #Example: Intel i5-14500, gen = 5
    if brand == "Intel" or brand == "AMD":
        nums = re.findall(r"\d+", model)
        if len(nums) == 2 and len(nums[1]) > 3:
            num = nums[1]
            return int(num[:2]) if len(num) >= 5 else int(num[:1])
    return None

#Extracts cpu tier from model
def extract_cpu_tier(model):
    brand = getModelBrandCPU(model)
    model = str(model).lower()

    #Cpu tier is handled differently in Apple side since they have labels instead of numbers
    if brand == "Apple":
        if "pro" in model:
            return 2
        elif "max" in model:
            return 3
        elif "ultra" in model:
            return 4
        else:
            return 1
    elif brand == "Intel" or brand == "AMD":
        nums = re.findall(r"\d+", model)

        #Get the last 3 digits of cpu model as cpu tier
        if len(nums) == 2 and len(nums[1]) > 3:
            num = nums[1]
            return int(num[2:]) if len(num) >= 5 else int(num[1:])
    return None

#Extracts gpu generation from model
def extract_gpu_gen(model):
    brand = getModelBrandGPU(model)
    model = str(model).lower()

    letters = "abcdef" #Indexer string for Intel GPUs
    if brand == "Apple": #Return 1 because the dataset only features "Apple Integrated" string as Apple GPU
        return 1
    elif brand == "NVIDIA" or brand == "AMD":
        nums = re.findall(r"\d+", model)

        #Take the digits except last 3 as generation
        if len(nums) != 0:
            num = nums[0]
            return int(num[:2]) if len(num) > 4 else int(num[0])
    elif brand == "Intel":
        parts = model.split()

        #For intel side, letters correspond to the GPU generation
        #Example: A = 1, B = 2, C = 3
        if len(parts) > 1 and parts[1][0] in letters:
            letter = parts[1][0]
            return int(letters.find(letter)) + 1 #Indexing in letters string
    return None

#Extracts gpu tier from model
def extract_gpu_tier(model):
    brand = getModelBrandGPU(model)
    model = str(model)

    if brand == "Apple": #Return 1 for Apple because there is no tier labels in this dataset
        return 1
    elif brand == "Intel":
        nums = re.findall(r"\d+", model)

        #Take first 2 digits in Intel as generation
        if len(nums) != 0 and len(nums[0]) > 1:
            num = nums[0]
            return int(num[:2])
    elif brand == "NVIDIA":
        nums = re.findall(r"\d+", model)

        #Take last 3 digits as tier
        if len(nums) != 0 and len(nums[0]) > 3:
            num = nums[0]
            return int(num[2:]) if len(num) > 4 else int(num[1:])
    elif brand == "AMD":
        nums = re.findall(r"\d+", model)

        #Take mid 2 digits as tier
        if len(nums) != 0 and len(nums[0]) > 3:
            num = nums[0]
            return int(num[2:4]) if len(num) > 4 else int(num[1:3])
    return None

#Extracts gpu premium feature from model
def extract_gpu_premium(model):
    brand = getModelBrandGPU(model)
    model = str(model).lower()

    #Check if GPU is a premium by label.
    if brand == "Intel" and "limited" in model:
        return 1
    elif brand == "AMD" and "xt" in model:
        return 1
    elif brand == "NVIDIA" and "ti" in model:
        return 1
    return 0

#Calculates cpu performance using family, generation and tier
def calculate_cpu_performance(model, family, gen, tier):
    brand = getModelBrandCPU(model)

    #Base family scores
    family_scores = {"i3": 30, "i5": 50, "i7": 70, "i9": 90,
                   "ryzen 3": 35, "ryzen 5": 55, "ryzen 7": 75, "ryzen 9": 95,
                   "apple m": 80}
    base = family_scores.get((family or "").lower(), 50) #Get the base score of model from family_scores table
    baseline_gen = {"Intel": 10, "AMD": 4, "Apple": 1} #Generation reference points, equivalent gens.
    brandMultiplier = {"Intel": 0.05, "AMD": 0.06, "Apple": 0.12} #Company price policy, for example Apple tends to make prices expensive
    tiersApple = {1: 100, 2: 400, 3: 700, 4:999} #3 digits are assigned for each Apple tier because other CPUs handled as 3 digits
    if gen is None:
        gen_multiplier = 1.0
    else:  
        delta = gen - baseline_gen.get(brand, 0) #Calculate deviation from generation reference point
        gen_effect = delta * brandMultiplier.get(brand, 0.05) #Generation deviation is multiplied by company price scale
        gen_effect = max(min(gen_effect, 0.35), -0.35)
        gen_multiplier = 1.0 + gen_effect

    if brand == "Apple":
        tier = tiersApple.get(tier, 0) #Convert Apple tier into a format which is same with other CPUs
    tier_score = (tier / 999.0) * 40.0 #Since tiers are 3 digits, scale it.
    cpu_perf = ((base + tier_score) * gen_multiplier) * 10 #Scale the score at the end
    return float(round(cpu_perf, 3))

#Calculates gpu performance using gpu generation and tiers
def calculate_gpu_performance(model, gen, tier, cpu_gen):
    brand = getModelBrandGPU(model)
    model = model.lower()

    if brand == "Apple":
        tier_major = tier
        tier_minor = 0
        gen = cpu_gen
    else:
        tier = str(tier)
        tier_major = int(tier[0])
        tier_minor = int(tier[1])

    base_tier_scores = {
        1: 10,
        2: 20,
        3: 35,
        4: 50,
        5: 70,
        6: 100,
        7: 125,
        8: 150,
        9: 180
    }
    base_tier_scores_apple = {
        1: 40,
        2: 80,
        3: 120,
        4: 150
    }
    #Get the base tier score based on brand
    if brand != "Apple":
        base = base_tier_scores.get(tier_major, 40)
    else:
        base = base_tier_scores_apple.get(tier_major, 40)

    baseline_gen = {"NVIDIA": 2, "AMD": 5, "Intel": 1, "Apple": 1} #Equivalent generation for each brand
    brandMultiplier = {"NVIDIA": 0.6, "AMD": 0.4, "Intel": 0.32, "Apple": 0.4} #Brand scores, for example NVIDIA produces the most powerful GPUs
    
    delta = gen - baseline_gen.get(brand, 1) #Generation deviation from reference point
    gen_multiplier = 1.0 + delta * brandMultiplier.get(brand, 0.05)

    #Premium label scores
    suffix_bonus = 1.0
    if "ti" in model:
        suffix_bonus = 1.15
    if "xtx" in model:
        suffix_bonus = 1.30
    elif "xt" in model:
        suffix_bonus = 1.18
    if "limited" in model:
        suffix_bonus = 1.075


    brand_bias = {"NVIDIA": 0, "AMD": -5, "Intel": -15, "Apple": -20}.get(brand, 0)
    gpu_perf = (base + tier_minor * 2 + brand_bias) * gen_multiplier
    gpu_perf *= suffix_bonus*10 #Scale the score at the end
    
    return float(round(gpu_perf, 3))
