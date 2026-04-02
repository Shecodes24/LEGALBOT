from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_bcrypt import Bcrypt
from database import get_db_connection  # Ensure this file exists and manages DB connections
from modules import create_tables       # Ensure this creates necessary DB tables
from werkzeug.utils import secure_filename
import os

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import pipeline


device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

import re
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

# Analytical model — better than BART for structured legal analysis
analyzer = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=device
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__, static_folder='static')
app.secret_key = "your_secret_key"
bcrypt = Bcrypt(app)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "my_vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional legal assistant specialized in the Indian Penal Code (IPC).
Answer ONLY using the provided context.
Be precise, factual, and concise."""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=groq_api_key
)

# Create RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

@app.route('/Chatbot')
def Chatbot():
    return render_template('Chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from RAG chain
        response = rag_chain.invoke(user_message)
        
        return jsonify({
            'response': response.content,
            'success': True
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500
# Ensure tables exist before running the app
create_tables()

# Route: Redirect to Login Page by Default
@app.route('/')
def home():
    return redirect(url_for('login_page'))

# Route: Serve Login Page
@app.route('/login_page')
def login_page():
    return render_template('login.html')

# Route: Serve Register Page
@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    phone = data.get('phone')
    country_code = data.get('country_code')
    password = data.get('password')
    role = data.get('role')
    dob = data.get('dob', None)

    if not all([username, email, phone, country_code, password, role]):
        return jsonify({"error": "All fields are required!"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if username or email already exists
        cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
        if cursor.fetchone():
            return jsonify({"error": "Username or Email already exists!"}), 400

        # Insert user with default NULL access (Pending Approval)
        cursor.execute(''' 
            INSERT INTO users (username, email, phone, country_code, password_hash, role, dob, access)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
        ''', (username, email, phone, country_code, hashed_password, role, dob))
        conn.commit()

        cursor.execute("SELECT LAST_INSERT_ID()")  # Get last inserted user ID
        user_id = cursor.fetchone()["LAST_INSERT_ID()"]

        print(f"✅ New User Registered: ID {user_id}, Pending Approval")

        # If user is a lawyer officer, insert into lawyer_users table
        if role == 'lawyer':
            lawyer_id = data.get('lawyer_id')
            station_name = data.get('station_name')
            lawyer_position = data.get('lawyer_position')

            if not all([lawyer_id, station_name, lawyer_position]):
                return jsonify({"error": "All lawyer details are required!"}), 400

            # Check if lawyer ID already exists
            cursor.execute("SELECT lawyer_id FROM lawyer_users WHERE lawyer_id = %s", (lawyer_id,))
            if cursor.fetchone():
                return jsonify({"error": "lawyer ID already exists!"}), 400

            cursor.execute('''
                INSERT INTO lawyer_users (id, lawyer_id, station_name, lawyer_position)
                VALUES (%s, %s, %s, %s)
            ''', (user_id, lawyer_id, station_name, lawyer_position))
            conn.commit()
            print(f"✅ lawyer User Registered: {lawyer_id}")

        # If user is a public user, insert into public_users table
        elif role == 'public':
            aadhaar = data.get('aadhaar')
            current_status = data.get('current_status')
            organization = data.get('organization')

            if not all([aadhaar, current_status]):
                return jsonify({"error": "Aadhaar and Current Status are required!"}), 400

            cursor.execute('''
                INSERT INTO public_users (id, aadhaar, current_status, organization)
                VALUES (%s, %s, %s, %s)
            ''', (user_id, aadhaar, current_status, organization))
            conn.commit()
            print(f"✅ Public User Registered: {aadhaar}")

        return jsonify({"message": "User registered successfully! Pending admin approval.", "redirect": url_for('login_page')}), 201

    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    finally:
        cursor.close()
        conn.close()

def summarygeneration(PDF_PATH):

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    full_text = "\n".join(doc.page_content for doc in docs)
    full_text = re.sub(r'\n+', ' ', full_text)
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    print("✅ PDF loaded successfully")
    print(f"Total pages: {len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(full_text)
    print(f"✅ Text split into {len(chunks)} chunks")

    # Per-chunk analytical dimensions
    section_results = {
        "parties": [],
        "obligations": [],
        "key_clauses": [],
        "risks": [],
        "facts": []
    }

    prompts = {
        "parties":     "Identify all parties, their roles and relationships in this legal text:\n\n",
        "obligations": "List the key obligations and responsibilities of each party in this legal text:\n\n",
        "key_clauses": "Identify the most important legal clauses or provisions in this text:\n\n",
        "risks":       "Identify legal risks, ambiguities, or problematic terms in this legal text:\n\n",
        "facts":       "Extract the key facts and background information from this legal text:\n\n"
    }

    for idx, chunk in enumerate(chunks):
        if len(chunk.split()) < 30:
            continue

        print(f"Analyzing chunk {idx + 1}/{len(chunks)}...")

        for dimension, prompt_prefix in prompts.items():
            try:
                input_text = prompt_prefix + chunk
                # flan-t5-large handles up to ~512 tokens; truncate if needed
                input_text = input_text[:1800]

                result = analyzer(
                    input_text,
                    max_new_tokens=180,
                    do_sample=False
                )
                output = result[0]["generated_text"].strip()

                if output and len(output) > 10:
                    section_results[dimension].append(output)

            except Exception as e:
                print(f"  Error in chunk {idx + 1} [{dimension}]: {e}")

    # Build final structured analytical report
    def merge_section(items, label):
        if not items:
            return f"### {label}\nNo information extracted.\n"
        combined = " | ".join(dict.fromkeys(items))  # deduplicate while preserving order
        return f"### {label}\n{combined}\n"

    final_summary = f"""
📄 LEGAL DOCUMENT ANALYTICAL REPORT
{'=' * 60}

{merge_section(section_results['parties'],     '👥 Parties & Roles')}
{merge_section(section_results['facts'],       '📋 Key Facts & Background')}
{merge_section(section_results['obligations'], '⚖️  Obligations & Responsibilities')}
{merge_section(section_results['key_clauses'], '📌 Important Clauses & Provisions')}
{merge_section(section_results['risks'],       '⚠️  Legal Risks & Ambiguities')}
"""

    print("\n" + "=" * 60)
    print(final_summary)
    return final_summary

@app.route("/summarization", methods=["GET", "POST"])
def summarization():
  
    summary = ""

    if request.method == "POST":
        file = request.files["pdf"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            summary = summarygeneration(file_path)
           

    return render_template(
        "summarization.html",
        summary=summary
    )

# Route: User Login (POST)
@app.route('/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check Admin Table First
        cursor.execute("SELECT * FROM admins WHERE username = %s", (username,))
        admin = cursor.fetchone()

        if admin and bcrypt.check_password_hash(admin['password_hash'], password):
            session['user_id'] = admin['id']
            session['role'] = 'admin'
            return jsonify({"message": "Admin login successful!", "redirect": url_for('admin_home')}), 200

        # Check Users Table
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and bcrypt.check_password_hash(user['password_hash'], password):
            # Check Access Control Before Granting Login
            if user['access'] is None:
                return jsonify({"error": "Your account is pending admin approval. Please wait."}), 403
            elif user['access'] == 'denied':
                return jsonify({"error": "Your account access has been denied by the admin."}), 403
            elif user['access'] == 'accepted':
                session['user_id'] = user['id']
                session['role'] = user['role']

                if user['role'] == 'admin':
                    return jsonify({"message": "Login successful!", "redirect": url_for('admin_home')}), 200
                elif user['role'] == 'lawyer':
                    return jsonify({"message": "Login successful!", "redirect": url_for('lawyer_home')}), 200
                elif user['role'] == 'public':
                    return jsonify({"message": "Login successful!", "redirect": url_for('public_home')}), 200

        return jsonify({"error": "Invalid username or password"}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Ensures JSON response even in case of an error

    finally:
        cursor.close()
        conn.close()


# Route: Admin Dashboard
@app.route('/admin_home')
def admin_home():
    # Fetch case statistics for the Admin Dashboard
    case_stats = get_case_stats_all()

    return render_template('admin_home.html', case_stats=case_stats)

# Route: Get Case Statistics (Total, Pending, Closed)
@app.route('/get_case_stats', methods=['GET'])
def get_case_stats():
    admin_id = session['user_id']
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) AS total FROM cases  where created_by= %s", (admin_id,))
        total = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) AS pending FROM cases WHERE status = 'Pending' and created_by= %s", (admin_id,))
        pending = cursor.fetchone()['pending']

        cursor.execute("SELECT COUNT(*) AS closed FROM cases WHERE status = 'Closed' and created_by= %s", (admin_id,))
        closed = cursor.fetchone()['closed']

        return {"total": total, "pending": pending, "closed": closed}

    except Exception as e:
        return {"error": str(e)}

    finally:
        cursor.close()
        conn.close()

# Route: Get Case Statistics (Total, Pending, Closed)
@app.route('/get_case_stats_all', methods=['GET'])
def get_case_stats_all():
  
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) AS total FROM cases")
        total = cursor.fetchone()['total']

        cursor.execute("SELECT COUNT(*) AS pending FROM cases WHERE status = 'Pending' ")
        pending = cursor.fetchone()['pending']

        cursor.execute("SELECT COUNT(*) AS closed FROM cases WHERE status = 'Closed'")
        closed = cursor.fetchone()['closed']

        return {"total": total, "pending": pending, "closed": closed}

    except Exception as e:
        return {"error": str(e)}

    finally:
        cursor.close()
        conn.close()

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/manage_cases', methods=['GET', 'POST'])
def manage_cases():
    case_id = request.args.get('case_id', type=int)
    connection = get_db_connection()
    cursor = connection.cursor()



    admin_id = session['user_id']

    if case_id:
        cursor.execute('SELECT * FROM cases WHERE id = %s', (case_id,))
        case = cursor.fetchone()

        cursor.execute('SELECT * FROM victims WHERE case_id = %s', (case_id,))
        victims = cursor.fetchall()

        cursor.execute('SELECT * FROM suspects WHERE case_id = %s', (case_id,))
        suspects = cursor.fetchall()

        cursor.execute('SELECT * FROM criminals WHERE case_id = %s', (case_id,))
        criminals = cursor.fetchall()
    else:
        case = None
        victims, suspects, criminals = [], [], []

    if request.method == 'POST':
        district = request.form['district']
        station = request.form.get('station', '').strip()
        fir_number = request.form['fir_number']
        incharge_case = request.form.get('incharge_case', '').strip()
        overview = request.form['case_overview']
        judgement = request.form.get('judgment', None)
        status = request.form['status']
        case_date = request.form['case_date']
        case_type = request.form['case_type']

        fir_file = request.files.get('fir_file', None)
        filename = None
        if fir_file and fir_file.filename:
            filename = secure_filename(fir_file.filename)
            fir_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        if case is None:
            cursor.execute('''
                INSERT INTO cases (district, station, fir_number, incharge_case, overview, judgement, status, fir_file, created_by, updated_by, case_date, case_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (district, station, fir_number, incharge_case, overview, judgement, status, filename, admin_id, admin_id, case_date, case_type))
            case_id = cursor.lastrowid
        else:
            cursor.execute('''
                UPDATE cases
                SET district = %s, station = %s, fir_number = %s, incharge_case = %s, overview = %s, judgement = %s, status = %s, fir_file = %s, updated_by = %s, case_date = %s, case_type = %s
                WHERE id = %s
            ''', (district, station, fir_number, incharge_case, overview, judgement, status, filename, admin_id, case_date, case_type, case_id))

        cursor.execute('DELETE FROM victims WHERE case_id = %s', (case_id,))
        cursor.execute('DELETE FROM suspects WHERE case_id = %s', (case_id,))
        cursor.execute('DELETE FROM criminals WHERE case_id = %s', (case_id,))

        victim_count = int(request.form['victim_count'])
        for i in range(1, victim_count + 1):
            name = request.form[f'victim_name_{i}']
            address = request.form[f'victim_address_{i}']
            phone = request.form[f'victim_phone_{i}']
            age = request.form[f'victim_age_{i}']
            aadhaar = request.form[f'victim_aadhaar_{i}']
            photo_file = request.files.get(f'victim_photo_{i}')
            photo_filename = None
            if photo_file and photo_file.filename:
                photo_filename = secure_filename(photo_file.filename)
                photo_file.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

            cursor.execute('''
                INSERT INTO victims (case_id, name, address, phone, age, aadhaar, photo)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (case_id, name, address, phone, age, aadhaar, photo_filename))

        suspect_count = int(request.form['suspect_count'])
        for i in range(1, suspect_count + 1):
            name = request.form[f'suspect_name_{i}']
            address = request.form[f'suspect_address_{i}']
            phone = request.form[f'suspect_phone_{i}']
            age = request.form[f'suspect_age_{i}']
            aadhaar = request.form[f'suspect_aadhaar_{i}']
            photo_file = request.files.get(f'suspect_photo_{i}')
            photo_filename = None
            if photo_file and photo_file.filename:
                photo_filename = secure_filename(photo_file.filename)
                photo_file.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

            cursor.execute('''
                INSERT INTO suspects (case_id, name, address, phone, age, aadhaar, photo)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (case_id, name, address, phone, age, aadhaar, photo_filename))

        criminal_count = int(request.form['criminal_count'])
        for i in range(1, criminal_count + 1):
            name = request.form[f'criminal_name_{i}']
            address = request.form[f'criminal_address_{i}']
            phone = request.form[f'criminal_phone_{i}']
            age = request.form[f'criminal_age_{i}']
            aadhaar = request.form[f'criminal_aadhaar_{i}']
            photo_file = request.files.get(f'criminal_photo_{i}')
            photo_filename = None
            if photo_file and photo_file.filename:
                photo_filename = secure_filename(photo_file.filename)
                photo_file.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

            cursor.execute('''
                INSERT INTO criminals (case_id, name, address, phone, age, aadhaar, photo)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (case_id, name, address, phone, age, aadhaar, photo_filename))

        connection.commit()
        return redirect(url_for('manage_cases'))

    connection.close()
    return render_template('manage_cases.html', case=case, victims=victims, suspects=suspects, criminals=criminals)

@app.route('/manage_users')
def manage_users():
    return render_template('manage_users.html')  # Replace with the actual template you want to use
@app.route('/get_users', methods=['GET'])
def get_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, username, email, phone, role, access FROM users")
        users = cursor.fetchall()
        return jsonify(users), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/update_access', methods=['POST'])
def update_access():
    data = request.json
    user_id = data.get('id')
    status = data.get('status')

    if not user_id or status not in ['accepted', 'denied']:
        return jsonify({"error": "Invalid request"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET access = %s WHERE id = %s", (status, user_id))
        conn.commit()
        return jsonify({"message": f"User access updated to {status}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/delete_user', methods=['DELETE'])
def delete_user():
    """Deletes a user by ID."""
    user_id = request.args.get('id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        return jsonify({"message": "User deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/view_reports', methods=['GET'])
def view_reports():
    """Displays all reported cases with victims, suspects, and criminals."""
  
    admin_id = session['user_id']
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    cursor.execute("SELECT id, fir_number, case_type, status,case_date,overview FROM cases where created_by= %s", (admin_id,))
    cases = cursor.fetchall()

    cases_with_details = []
    for case in cases:
        case_id = case['id']

        cursor.execute("SELECT name, phone FROM victims WHERE case_id = %s", (case_id,))
        victims = cursor.fetchall()

        cursor.execute("SELECT name, phone FROM suspects WHERE case_id = %s", (case_id,))
        suspects = cursor.fetchall()

        cursor.execute("SELECT name, phone FROM criminals WHERE case_id = %s", (case_id,))
        criminals = cursor.fetchall()

        cases_with_details.append({
            "id": case_id,
            "fir_number": case['fir_number'],
            "case_type": case['case_type'],
            "status": case['status'],
            "victims": victims,
            "suspects": suspects,
            "criminals": criminals,
            "Schdule_date":case['case_date'],
            "case_overview":case['overview'],
        })

    connection.close()
    
    return render_template('view_reports.html', cases=cases_with_details)

@app.route('/edit_case/<int:case_id>', methods=['GET', 'POST'])
def edit_case(case_id):
  
    connection = get_db_connection()
    cursor = connection.cursor()
    
    cursor.execute("SELECT * FROM cases WHERE id = %s FOR UPDATE", (case_id,))
    case = cursor.fetchone()

    if not case:
        return "Error: Case not found", 404

    if request.method == 'POST':
        try:
            district = request.form.get('district', '').strip()
            station = request.form.get('station', '').strip()
            fir_number = request.form.get('fir_number', '').strip()
            incharge_case = request.form.get('incharge_case', '').strip()
            overview = request.form.get('case_overview', '').strip()
            judgement = request.form.get('judgment', '').strip()
            status = request.form.get('status', '').strip()
            case_date = request.form.get('case_date', '').strip()
            case_type = request.form.get('case_type', '').strip()

            fir_file = request.files.get('fir_file')
            filename = case['fir_file']   # default to existing file

            if fir_file and fir_file.filename:
                filename = secure_filename(fir_file.filename)
                fir_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Update case table
            cursor.execute('''
                UPDATE cases
                SET district = %s, station = %s, fir_number = %s, incharge_case = %s, 
                    overview = %s, judgement = %s, status = %s, fir_file = %s, 
                    case_date = %s, case_type = %s
                WHERE id = %s
            ''', (district, station, fir_number, incharge_case, overview, judgement,
                  status, filename, case_date, case_type, case_id))

            # Wipe and re-insert related entities
            cursor.execute("DELETE FROM victims WHERE case_id = %s", (case_id,))
            cursor.execute("DELETE FROM suspects WHERE case_id = %s", (case_id,))
            cursor.execute("DELETE FROM criminals WHERE case_id = %s", (case_id,))

            # Re-insert victims
            victim_count = int(request.form['victim_count'])
            for i in range(1, victim_count + 1):
                name = request.form[f'victim_name_{i}']
                address = request.form[f'victim_address_{i}']
                phone = request.form[f'victim_phone_{i}']
                age = request.form[f'victim_age_{i}']
                aadhaar = request.form[f'victim_aadhaar_{i}']
                photo_file = request.files.get(f'victim_photo_{i}')
                photo_filename = None
                if photo_file and photo_file.filename:
                    photo_filename = secure_filename(photo_file.filename)
                    photo_file.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

                cursor.execute('''
                    INSERT INTO victims (case_id, name, address, phone, age, aadhaar, photo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (case_id, name, address, phone, age, aadhaar, photo_filename))

            # Re-insert suspects
            suspect_count = int(request.form['suspect_count'])
            for i in range(1, suspect_count + 1):
                name = request.form[f'suspect_name_{i}']
                address = request.form[f'suspect_address_{i}']
                phone = request.form[f'suspect_phone_{i}']
                age = request.form[f'suspect_age_{i}']
                aadhaar = request.form[f'suspect_aadhaar_{i}']
                photo_file = request.files.get(f'suspect_photo_{i}')
                photo_filename = None
                if photo_file and photo_file.filename:
                    photo_filename = secure_filename(photo_file.filename)
                    photo_file.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

                cursor.execute('''
                    INSERT INTO suspects (case_id, name, address, phone, age, aadhaar, photo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (case_id, name, address, phone, age, aadhaar, photo_filename))

            # Re-insert criminals
            criminal_count = int(request.form['criminal_count'])
            for i in range(1, criminal_count + 1):
                name = request.form[f'criminal_name_{i}']
                address = request.form[f'criminal_address_{i}']
                phone = request.form[f'criminal_phone_{i}']
                age = request.form[f'criminal_age_{i}']
                aadhaar = request.form[f'criminal_aadhaar_{i}']
                photo_file = request.files.get(f'criminal_photo_{i}')
                photo_filename = None
                if photo_file and photo_file.filename:
                    photo_filename = secure_filename(photo_file.filename)
                    photo_file.save(os.path.join(app.config['UPLOAD_FOLDER'], photo_filename))

                cursor.execute('''
                    INSERT INTO criminals (case_id, name, address, phone, age, aadhaar, photo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (case_id, name, address, phone, age, aadhaar, photo_filename))

            connection.commit()
            return redirect(url_for('view_reports'))

        except Exception as e:
            connection.rollback()
            print(f"❌ Error updating case: {e}")

    # GET request — load existing data
    cursor.execute("SELECT * FROM victims WHERE case_id = %s", (case_id,))
    victims = cursor.fetchall()

    cursor.execute("SELECT * FROM suspects WHERE case_id = %s", (case_id,))
    suspects = cursor.fetchall()

    cursor.execute("SELECT * FROM criminals WHERE case_id = %s", (case_id,))
    criminals = cursor.fetchall()

    connection.close()
    return render_template('edit_cases.html', case=case,
                           victims=victims, suspects=suspects, criminals=criminals)

# Route: Homepages for different roles
@app.route('/lawyer_home')
def lawyer_home():
    return render_template('lawyer_home.html')

@app.route('/cases', methods=['GET'])
def cases():
    """Displays all reported cases with victims, suspects, and criminals."""
    
    admin_id = session['user_id']
    
    connection = get_db_connection()
    cursor = connection.cursor()
   
    
    cursor.execute("SELECT id, fir_number, case_type, status FROM cases where created_by= %s", (admin_id,))
    cases = cursor.fetchall()

    cases_with_details = []
    for case in cases:
        case_id = case['id']

        cursor.execute("SELECT name, phone FROM victims WHERE case_id = %s", (case_id,))
        victims = cursor.fetchall()

        cursor.execute("SELECT name, phone FROM suspects WHERE case_id = %s", (case_id,))
        suspects = cursor.fetchall()

        cursor.execute("SELECT name, phone FROM criminals WHERE case_id = %s", (case_id,))
        criminals = cursor.fetchall()

        cases_with_details.append({
            "id": case_id,
            "fir_number": case['fir_number'],
            "case_type": case['case_type'],
            "status": case['status'],
            "victims": victims,
            "suspects": suspects,
            "criminals": criminals
        })

    connection.close()
    
    return render_template('cases.html', cases=cases_with_details)

@app.route('/view_case/<int:case_id>')
def view_case(case_id):
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT * FROM cases WHERE id = %s", (case_id,))
        case = cursor.fetchone()

        if not case:
            return "Case not found", 404

        cursor.execute("SELECT * FROM victims WHERE case_id = %s", (case_id,))
        victims = cursor.fetchall()

        cursor.execute("SELECT * FROM suspects WHERE case_id = %s", (case_id,))
        suspects = cursor.fetchall()

        cursor.execute("SELECT * FROM criminals WHERE case_id = %s", (case_id,))
        criminals = cursor.fetchall()

        return render_template('view_case.html', case=case, victims=victims, suspects=suspects, criminals=criminals)

    finally:
        cursor.close()
        connection.close()



@app.route('/view_details')
def view_details():
    if 'user_id' not in session or session['role'] != 'lawyer':
        return redirect(url_for('login_page'))

    query = request.args.get('query', '').lower()
    connection = get_db_connection()
    cursor = connection.cursor()

    # Search logic (simple OR across all types)
    cursor.execute("""
        SELECT id, name, address, phone, age, aadhaar, photo, 'victim' AS role
        FROM victims
        WHERE LOWER(name) LIKE %s OR aadhaar LIKE %s OR phone LIKE %s
        UNION
        SELECT id, name, address, phone, age, aadhaar, photo, 'suspect' AS role
        FROM suspects
        WHERE LOWER(name) LIKE %s OR aadhaar LIKE %s OR phone LIKE %s
        UNION
        SELECT id, name, address, phone, age, aadhaar, photo, 'criminal' AS role
        FROM criminals
        WHERE LOWER(name) LIKE %s OR aadhaar LIKE %s OR phone LIKE %s
    """, tuple(['%' + query + '%'] * 9))

    results = cursor.fetchall()
    connection.close()

    return render_template('view_details.html', results=results)

@app.route('/view_profile/<role>/<int:person_id>', methods=['GET'])
def view_profile(role, person_id):
    # Check if user is logged in and has the 'lawyer' role
    if 'user_id' not in session or session['role'] != 'lawyer':
        return redirect(url_for('login_page'))

    connection = get_db_connection()
    cursor = connection.cursor()

    # Determine the table based on the role
    table = None
    if role == 'victim':
        table = 'victims'
    elif role == 'suspect':
        table = 'suspects'
    elif role == 'criminal':
        table = 'criminals'
    else:
        return "Invalid role", 400

    # Fetch the person's details from the appropriate table
    cursor.execute(f"SELECT * FROM {table} WHERE id = %s", (person_id,))
    person = cursor.fetchone()

    if not person:
        return "Person not found", 404

    # Fetch cases associated with this person (based on case_id)
    cursor.execute('''
    SELECT c.id AS case_id, c.fir_number, c.incharge_case, c.status, c.case_date, c.judgement, c.overview, c.case_type, c.district, c.station
    FROM cases c
    JOIN (
        SELECT case_id FROM victims WHERE id = %s
        UNION
        SELECT case_id FROM suspects WHERE id = %s
        UNION
        SELECT case_id FROM criminals WHERE id = %s
    ) AS case_ids
    ON c.id = case_ids.case_id
''', (person_id, person_id, person_id))

    # Fetch all the cases involving the person
    cases = cursor.fetchall()

    connection.close()

    return render_template('view_profile.html', person=person, role=role, cases=cases)

@app.route('/public_home')
def public_home():
    return render_template('public_home.html')

@app.route('/safety_tips')
def safety_tips():
    return render_template('safety_tips.html')

@app.route('/view_public_reports')
def view_public_reports():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT case_type, overview, judgement FROM cases")
    cases = cursor.fetchall()

    conn.close()
    return render_template('view_simple_reports.html', cases=cases)


# Route: Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    app.run(debug=True)
