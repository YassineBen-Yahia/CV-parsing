import random
from typing import List, Tuple, Dict
import re
from faker import Faker
faker = Faker()

# Sample data for augmentation
SKILLS = ["Python", "Machine Learning", "Data Analysis", "Project Management", "SQL", "JavaScript", "Communication"]
DEGREES = ["B.Sc. Computer Science", "M.Sc. Data Science", "MBA", "Ph.D. Physics", "B.A. Economics"]
POSITIONS = ["Software Engineer", "Data Scientist", "Project Manager", "Business Analyst", "Researcher"]
COMPANIES = ["Google", "Microsoft", "Amazon", "Facebook", "IBM", "Deloitte"]
DURATIONS = ["2 years", "Jan 2020 - Dec 2022", "2018-2021", "3 months", "since 2019"]

# Entity labels for spaCy
LABELS = {
    "SKILL": SKILLS,
    "DEGREE": DEGREES,
    "POSITION": POSITIONS,
    "COMPANY": COMPANIES,
    "DURATION": DURATIONS
}

def generate_resume_sentence() -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Generate a synthetic resume sentence and its entity annotations.
    Returns:
        text (str): The generated sentence.
        entities (List[Tuple[int, int, str]]): List of (start, end, label) for entities.
    """
    skill = random.choice(SKILLS)
    degree = random.choice(generate_degree_variations([]))
    position = random.choice(generate_designation_variations([]))
    company = random.choice(generate_company_variations([]))
    duration = random.choice(DURATIONS)
    
    templates = [
        f"Worked as a {position} at {company} for {duration} with expertise in {skill}.",
        f"{degree} holder, previously employed at {company} as a {position} for {duration}. Skills include {skill}.",
        f"{position} at {company} ({duration}), skilled in {skill}. Degree: {degree}.",
        f"{company} {position} ({duration}), {degree}, specializes in {skill}."
    ]
    text = random.choice(templates)
    
    # Find entity spans
    entities = []
    for label, values in LABELS.items():
        for value in values:
            start = text.find(value)
            if start != -1:
                end = start + len(value)
                entities.append((start, end, label))
    return text, entities


def create_spacy_training_data(n_samples: int = 100) -> List[Dict]:
    """
    Create spaCy training data with n_samples synthetic examples.
    Returns:
        List of dicts with 'text' and 'entities'.
    """
    data = []
    for _ in range(n_samples):
        text, entities = generate_resume_sentence()
        data.append({"text": text, "entities": entities})
    return data


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
        "Unix", "Windows Server", "Networking", "Security", "A/B Testing"
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


if __name__ == "__main__":
    n = 100  # Number of samples to generate
    training_data = create_spacy_training_data(n)
    # Save to file in spaCy format
    import json
    with open("synthetic_spacy_resume_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print(f"Generated {n} synthetic resume samples for spaCy training.")
    print("fonction bechir:", generate_designation_variations(["Software Engineer", "Data Scientist", "Project Manager", "Business Analyst", "Researcher"]))
