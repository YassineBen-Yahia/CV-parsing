import random
from typing import List, Tuple, Dict
from tools import filter_non_overlapping_spans
import re
from faker import Faker
import json
import logging
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
faker = Faker()



def augment_and_balance_data(train_data):
    """
    Apply data augmentation and balancing techniques to the training data.
    Focus on improving recall for entities like Designation, Companies worked at, and Degree.
    """
    logging.info("Starting data augmentation and balancing...")
    nlp = spacy.blank("en")
    
    # Analyze current entity distribution
    entity_counts = count_entities_by_type(train_data)
    logging.info(f"Initial entity distribution: {entity_counts}")
    
    # Augment entities with low recall
    augmented_entities = augment_entities(train_data, factor=3)
    
    # Add augmented entities to training data
    train_data.extend(augmented_entities)
    
    # Generate synthetic examples focused on skills
    skills = extract_entities_of_type(train_data, "Skills")
    synthetic_skills_examples = generate_synthetic_skills_examples(skills, count=100)
    train_data.extend(synthetic_skills_examples)
    
    # Introduce look-alike non-entities to improve precision
    """lookalikes = []
    for _ in range(50):
        lookalikes.append((
            generate_email_lookalike(),
            {"entities": []}
        ))
        lookalikes.append((
            generate_college_lookalike(),
            {"entities": []}
        ))
    train_data.extend(lookalikes)"""
    
    # Shuffle the training data to mix original and augmented examples
    random.shuffle(train_data)
    
    # Final entity distribution after augmentation
    final_entity_counts = count_entities_by_type(train_data)
    logging.info(f"Final entity distribution after augmentation: {final_entity_counts}")
    
    logging.info(f"Total training samples after augmentation: {len(train_data)}")


    random.shuffle(train_data)
    
    train_path = r"C:\ML\CV-Parsing\Data\augmented_training_data.spacy"
    
    doc_bin = DocBin()
    for text, entities in tqdm(train_data, desc=f"Processing {train_path}"):
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is not None:
                spans.append(span)
        filtered_spans = filter_non_overlapping_spans(spans)
        doc.ents = filtered_spans
        doc_bin.add(doc)
    doc_bin.to_disk(train_path)
    print(f"Saved augmented train to {train_path}")

    with open(r"C:\ML\CV-Parsing\Data\augmented_train_data.json", "w", encoding="utf-8") as f:
        json.dump([{"text": text, "entities": [[s, e, l] for s, e, l in anns["entities"]]} 
            for text, anns in train_data], f, indent=2)
    
    return train_data




def augment_entities(data, factor=3):
    """
    Focus on improving recall for entities like Designation, Companies worked at, and Degree
    by generating more diverse examples and contextual variations.
    """
    logging.info("Augmenting entities with low recall (Designation, Companies worked at, Degree)")
    augmented = []
    
    # Extract all examples of these entity types
    designations = extract_entities_of_type(data, "Designation")
    companies = extract_entities_of_type(data, "Companies worked at")
    degrees = extract_entities_of_type(data, "Degree")
    skills= extract_entities_of_type(data, "Skills")
    
    # Generate position title variations
    designation_variations = generate_designation_variations(designations)
    
    # Generate company name variations
    company_variations = generate_company_variations(companies)
    
    # Generate degree variations
    degree_variations = generate_degree_variations(degrees)

    # Expand skills vocabulary
    skills_variations = expand_skills_vocabulary(skills)
    
    # Apply entity swapping and contextual enrichment
    for text, annotations in tqdm(data, desc="Augmenting entities"):
        entities = annotations["entities"]
        has_target_entity = any(label in ["Designation", "Companies worked at", "Degree", "Skills"] 
                               for _, _, label in entities)
        
        if not has_target_entity:
            continue
            
        # Create multiple augmented versions
        for _ in range(factor):
            aug_text = text
            aug_entities = []
            offset = 0
            
            # Sort entities by position
            sorted_entities = sorted(entities, key=lambda x: x[0])
            
            for start, end, label in sorted_entities:
                entity_text = text[start:end]
                
                # Generate variations for target entity types
                if label == "Designation":
                    if random.random() < 0.8:  # 80% chance to substitute
                        new_designation = random.choice(designation_variations)
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_designation + after
                        
                        new_offset = offset + len(new_designation) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_designation), label))
                        offset = new_offset
                        continue
                
                elif label == "Companies worked at":
                    if random.random() < 0.7:  # 70% chance to substitute
                        new_company = random.choice(company_variations)
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_company + after
                        
                        new_offset = offset + len(new_company) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_company), label))
                        offset = new_offset
                        continue
                        
                elif label == "Degree":
                    if random.random() < 0.8:  # 80% chance to substitute
                        new_degree = random.choice(degree_variations)
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_degree + after
                        
                        new_offset = offset + len(new_degree) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_degree), label))
                        offset = new_offset
                        continue
                elif label == "Skills":
                    if random.random() < 0.6:  # 60% chance to substitute
                        new_skill = random.choice(skills_variations)
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_skill + after
                        
                        new_offset = offset + len(new_skill) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_skill), label))
                        offset = new_offset
                        continue
                
                # For other entities, keep them as is
                aug_entities.append((start + offset, end + offset, label))
            
            augmented.append((aug_text, {"entities": aug_entities}))
            
    # Generate synthetic examples with rich context for these entities


    #synthetic_examples = generate_synthetic_context_examples(factor * 2)
    #augmented.extend(synthetic_examples)
    
    logging.info(f"Created {len(augmented)} examples of entities")
    return augmented







def generate_designation_variations(designations):
    """
    Generate variations of job titles and designations to improve recall.
    """
    base_titles = set()
    for designation in designations:
        # Extract root title
        base = re.sub(r'^(Senior|Junior|Lead|Principal|Chief|Associate|Assistant)\s+', '', designation)
        base = re.sub(r'\s+(I|II|III|IV|V)$', '', base)
        if len(base) > 3:  # Avoid too short titles
            base_titles.add(base)
    
    variations = []
    prefixes = ['Senior', 'Junior', 'Lead', 'Principal', 'Chief', 'Associate', 'Assistant', '']
    suffixes = [' I', ' II', ' III', '', ' Manager', ' Lead']
    
    for base in base_titles:
        for prefix in prefixes:
            for suffix in suffixes:
                variation = f"{prefix} {base}{suffix}".strip()
                if variation and variation != base:
                    variations.append(variation)
    
    # Add completely new designations
    additional_titles = [
        "Machine Learning Engineer", "Cloud Architect", "DevOps Specialist",
        "AI Researcher", "Data Engineer", "Blockchain Developer",
        "Frontend Engineer", "Backend Developer", "Full Stack Engineer",
        "Site Reliability Engineer", "UX Designer", "UI Developer",
        "Product Owner", "Scrum Master", "Technical Program Manager",
        "Solutions Architect", "Systems Analyst", "Network Administrator",
        "Information Security Analyst", "Database Administrator"
    ]
    
    variations.extend(additional_titles)
    
    # Add fake but realistic titles from Faker
    for _ in range(50):
        variations.append(faker.job())
    
    return list(set(variations))


def generate_company_variations(companies):
    """
    Generate company name variations to improve recall.
    """
    variations = []
    suffixes = [' Inc.', ' LLC', ' Ltd.', ' Corporation', ' Corp.', ' Company', 
                ' Technologies', ' Group', ' Solutions', ' International', '']
    
    for company in companies:
        # Remove existing suffix if any
        base = re.sub(r'\s+(Inc|LLC|Ltd|Corporation|Corp|Company|Technologies|Group|Solutions|International)\.?$', '', company)
        
        # Add different suffixes
        for suffix in suffixes:
            if not company.endswith(suffix):
                variation = f"{base}{suffix}".strip()
                if variation and variation != company:
                    variations.append(variation)
    
    # Add completely new company names
    tech_companies = [
        "Quantum Computing", "Neural Dynamics", "Cloud Solutions",
        "Data Insights", "Blockchain Innovations", "Tech Frontiers",
        "Digital Transformation", "AI Systems", "Smart Analytics",
        "Future Technologies", "Cyber Security Solutions", "Virtual Systems",
        "Global Software", "Mobile Innovations", "Enterprise Solutions"
    ]
    
    for company in tech_companies:
        for suffix in suffixes:
            variations.append(f"{company}{suffix}".strip())
    
    # Add fake but realistic company names from Faker
    for _ in range(50):
        variations.append(faker.company())
    
    return list(set(variations))


def generate_degree_variations(degrees):
    """
    Generate degree variations to improve recall.
    """
    variations = []
    
    # Common degree types and their variations
    degree_types = {
        "Bachelor": ["Bachelor of", "Bachelor's in", "Bachelor's degree in", "B.S. in", "B.A. in", "BS in", "BA in"],
        "Master": ["Master of", "Master's in", "Master's degree in", "M.S. in", "M.A. in", "MS in", "MA in"],
        "PhD": ["PhD in", "Ph.D. in", "Doctorate in", "Doctoral degree in"],
        "Associate": ["Associate of", "Associate's in", "A.S. in", "A.A. in"]
    }
    
    # Common fields of study
    fields = [
        "Computer Science", "Information Technology", "Software Engineering", 
        "Data Science", "Artificial Intelligence", "Business Administration",
        "Information Systems", "Electrical Engineering", "Computer Engineering",
        "Mathematics", "Statistics", "Economics", "Finance", "Marketing",
        "Management", "Human Resources", "Psychology", "Communications"
    ]
    
    # Generate variations
    for degree_type, variations_list in degree_types.items():
        for variation in variations_list:
            for field in fields:
                variations.append(f"{variation} {field}")
    
    # Add existing degrees with slight modifications
    for degree in degrees:
        # Try different abbreviations and formatting
        degree = degree.replace("Bachelor of", "B.S. in")
        degree = degree.replace("Master of", "M.S. in")
        variations.append(degree)
        
        # Add "honors" or other qualifiers
        if "Bachelor" in degree or "B.S." in degree or "B.A." in degree:
            variations.append(f"{degree} with Honors")
            variations.append(f"{degree} (Honours)")
    
    return list(set(variations))


def expand_skills_vocabulary(existing_skills):
    """
    Expand skills vocabulary with variations and additional skills.
    """
    expanded = set(existing_skills)
    
    # Technical skills
    programming_languages = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "Swift", "Kotlin", "PHP", "Ruby", "Scala", "R", "MATLAB", "Perl", "Shell"
    ]
    
    web_technologies = [
        "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django",
        "Flask", "Spring Boot", "ASP.NET", "jQuery", "Bootstrap", "Tailwind CSS", 
        "GraphQL", "REST API", "SOAP", "WebSockets"
    ]
    
    databases = [
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "SQL Server",
        "Redis", "Cassandra", "DynamoDB", "Firebase", "Neo4j", "Elasticsearch"
    ]
    
    cloud_technologies = [
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform",
        "Jenkins", "GitLab CI/CD", "GitHub Actions", "Ansible", "Puppet", "Chef",
        "Serverless", "Lambda", "S3", "EC2", "Azure Functions", "Cloud Run"
    ]
    
    data_science = [
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Analysis",
        "Statistical Modeling", "TensorFlow", "PyTorch", "scikit-learn", "pandas",
        "NumPy", "SciPy", "Matplotlib", "Tableau", "Power BI", "Big Data", "Hadoop",
        "Spark", "Airflow", "Jupyter", "Neural Networks"
    ]
    
    other_tech = [
        "Git", "Agile", "Scrum", "Kanban", "Jira", "Confluence", "DevOps", "CI/CD",
        "Test-Driven Development", "Microservices", "RESTful API", "OOP", "Linux",
        "Unix", "Windows Server", "Networking", "Security", "A/B Testing" , "Excel" , "UI/UX Design"
        , "Prototyping", "Figma", "Adobe XD" ,"R"
    ]
    
    # Add all these skills
    expanded.update(programming_languages)
    expanded.update(web_technologies)
    expanded.update(databases)
    expanded.update(cloud_technologies)
    expanded.update(data_science)
    expanded.update(other_tech)
    
    # Create variations with frameworks and tools
    variations = []
    for skill in ["Python", "JavaScript", "Java", "C#"]:
        if skill == "Python":
            variations.extend([
                "Python Django", "Python Flask", "Python FastAPI", "Python Pandas",
                "Python scikit-learn", "Python Data Analysis", "Python Automation"
            ])
        elif skill == "JavaScript":
            variations.extend([
                "JavaScript React", "JavaScript Node.js", "JavaScript Angular",
                "JavaScript Vue", "JavaScript Express", "JavaScript Front-end"
            ])
        elif skill == "Java":
            variations.extend([
                "Java Spring", "Java Hibernate", "Java J2EE", "Java Android",
                "Java Microservices", "Java Backend Development"
            ])
        elif skill == "C#":
            variations.extend([
                "C# .NET", "C# ASP.NET", "C# Unity", "C# WPF", "C# Xamarin",
                "C# Entity Framework"
            ])
    
    expanded.update(variations)
    
    # Add specific versions
    version_variations = []
    for skill in ["Python", "Java", "JavaScript", "React", "Angular"]:
        if skill == "Python":
            version_variations.extend(["Python 3.8", "Python 3.9", "Python 3.10"])
        elif skill == "Java":
            version_variations.extend(["Java 11", "Java 17", "Java 8"])
        elif skill == "JavaScript":
            version_variations.extend(["JavaScript ES6", "JavaScript ES2022"])
        elif skill == "React":
            version_variations.extend(["React 16", "React 17", "React 18"])
        elif skill == "Angular":
            version_variations.extend(["Angular 12", "Angular 13", "Angular 14"])
    
    expanded.update(version_variations)
    
    return list(expanded)

def generate_synthetic_skills_examples(skills, count=50):
    """
    Generate synthetic examples focused on skills.
    """
    examples = []
    
    templates = [
        "Technical Skills:\n{skills}",
        "Proficient in the following technologies: {skills}",
        "Key skills include: {skills}",
        "Technologies: {skills}",
        "Programming Languages & Frameworks: {skills}",
        "Technical Expertise: {skills}",
        "Skills & Competencies: {skills}"
    ]
    
    for _ in range(count):
        # Select random skills
        selected_skills = random.sample(skills, k=20) if len(skills) >= 20 else skills
        
        # Choose a template
        template = random.choice(templates)
        
        # Choose a separator
        separator = random.choice([", ", " | ", "; ", "\n- ", ", and "])
        
        # Create the skills list
        skills_list = separator.join(selected_skills)
        
        # Generate text
        text = template.replace("{skills}", skills_list)
        
        # Create entities list
        entities = []
        for skill in selected_skills:
            skill_pos = text.find(skill)
            if skill_pos >= 0:
                entities.append((skill_pos, skill_pos + len(skill), "Skills"))
        
        if entities:
            examples.append((text, {"entities": entities}))
    
    return examples





def generate_email_lookalike():
    """
    Generate text that looks like email but isn't valid.
    """
    almost_emails = [
        "john.doe[at]gmail.com",
        "contact-us(example.com)",
        "email: info@company",
        "www.example.com/contact",
        "user@localhost",
        "firstname_lastname@",
        "info(at)company.com",
        "email-address.com"
    ]
    return random.choice(almost_emails)

def generate_college_lookalike():
    """
    Generate text that looks like college name but shouldn't be labeled as one.
    """
    almost_colleges = [
        "Company University Program",
        "Educational Resources Inc",
        "The Learning Center",
        "Professional Academy",
        "Training Institute",
        "Certification Program",
        "Corporate University",
        "The Knowledge Hub"
    ]
    return random.choice(almost_colleges)


def extract_entities_of_type(data, entity_type):
    """
    Extract all instances of a specific entity type from the data.
    """
    entities = []
    for text, annotations in data:
        for start, end, label in annotations["entities"]:
            if label == entity_type:
                entities.append(text[start:end])
    return entities


def count_entities_by_type(data):
    """
    Count entities by type in the data.
    """
    counts = {}
    for text, annotations in data:
        for start, end, label in annotations["entities"]:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
    return counts


"""if __name__ == "__main__":
    n = 100  # Number of samples to generate
    import json
    from Data_loader import load_and_export_ner_data
    train_data, val_data = load_and_export_ner_data()
    augmented_train = augment_and_balance_data(train_data)"""