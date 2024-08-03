import csv
import os
from tqdm import tqdm
from Bio import PDB
import wget
import ssl
from Bio.PDB import PDBParser,PDBIO
import shutil

def is_valid_pdb(file_path):
    try:
        parser = PDBParser(QUIET=True)
        parser.get_structure("protein", file_path)
        return True
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return False

def check_pdb_files(directory):
    valid_pdbs = []
    invalid_pdbs = []

    for filename in os.listdir(directory):
        if not filename.endswith(".pdb"):
            invalid_pdbs.append(filename)
            continue
        file_path = os.path.join(directory, filename)
        if is_valid_pdb(file_path):
            valid_pdbs.append(filename)
        else:
            invalid_pdbs.append(filename)

    return valid_pdbs, invalid_pdbs     

def count_residues(pdb_file):
    """
    Method for counting the number residues in a given pdb-file

    Args:

    - pdb_file (str): Path to pdb_file.

    Returns:

    - the number of residues in the pdb_file 
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    residue_count = 0

    for model in structure:
        for chain in model:
            residue_count += len(chain)

    return residue_count
def format_domain_annotation_file_chainsaw_discMerge(chainsaw_annotation_tsv, output_dir):
    """
    Formats the CHAINSAW domain annotation output file into a specified domain-annotation format,
    discontinuous domains are merged into a single domain from the start of the first
    segment to the end of the last segment, ignoring included domains. All domain annotations are treated as chain_A domain annotations!
    
    Args:
    - chainsaw_annotation_tsv (str): Path to the CHAINSAW output domain annotation file.
    - output_dir (str): Path to a directory where the formatted domain-annotation file will be saved.
    

    Returns:
    - The path to the formatted domain annotation file if successful, otherwise None.
    """
    with open(chainsaw_annotation_tsv, 'r', newline='', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        header = next(tsv_reader)
        chain_id_index = header.index("chain_id")
        chopping_index = header.index("chopping")
        formatted_domains = []
        last_end = 0  
        for row in tsv_reader:
            if row[chopping_index] == "NULL":
                print("0 domains found in CHAINSAW annotation")
                return None
            chain_id = row[chain_id_index]
            domains = row[chopping_index].split(',')
            for domain in domains:
                segments = domain.split('_')
                start = int(segments[0].split('-')[0])
                end = int(segments[-1].split('-')[-1])
                
                if start < last_end :
                    continue
            
                formatted_domains.append((chain_id, start, end))
                last_end = end
            
    directory, file_name = os.path.split(chainsaw_annotation_tsv)
    new_file_name = file_name.replace('.tsv', '_prot2DFormattedDomains.tsv')
    domain_annotation_formatted_file = os.path.join(output_dir, new_file_name)

    with open(domain_annotation_formatted_file, 'w', newline='', encoding='utf-8') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(['chain_id', 'domain', 'domain_start', 'domain_end'])
        for idx, (chain_id, start, end) in enumerate(formatted_domains, start=1):
            tsv_writer.writerow([chain_id+"_chain_A", idx, start, end])
    return domain_annotation_formatted_file
def clear_dir(directory):
    for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
def check_and_delete_directory(directory):
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    if os.path.exists(directory) and os.listdir(directory):
        user_input = input(f"\n{YELLOW}The directory {directory} is not empty. Do you want to delete its contents? [Y/N]: {RESET}").strip().upper()
        if user_input == 'Y':
            # Delete the contents of the directory
            clear_dir(directory)
        else:
            print("Operation aborted.")
            return False
    return True

def read_in_foldseek_cluster_tsv(cluster_tsv_file):
    clusters_dict = {}
    with open(cluster_tsv_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            representative = row[0]
            member = row[1]
            if representative not in clusters_dict:
                clusters_dict[representative] = 0
            clusters_dict[representative] += 1
    return clusters_dict

def format_domain_annotation_file_chainsaw_discSplit(chainsaw_annotation_tsv, output_dir):
    """
    Can be used for formatting chainsaw (domain annotation software) output file into the here needed domain-annotatino format.
    Discontinous domains are split into seperate domains. All domain annotations are treated as chain_A domain annotations!
    Args:

    - chainsaw_annotation_tsv (str): Path to chainsaw output domain annotation file.
    - output_dir (str): Path to a directory where the formatted domain-annotaion file will be saved.

    Returns:

    - The path to the formatted domain annotation file if successfull. Otherwise None.
    """
    
    with open(chainsaw_annotation_tsv, 'r', newline='', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        header = next(tsv_reader)
        chain_id_index = header.index("chain_id")
        chopping_index = header.index("chopping")
        
        #Extract chain ID (assuming it's the same for all rows)
        chain_id = None
        for row in tsv_reader:
            if row[chopping_index] == "NULL":
                print("0 domains found in chainsaw annotation")
                return None
            chain_id = row[chain_id_index]
            row[chopping_index]=row[chopping_index].replace('_',',')
            string_domains = row[chopping_index].split(',')
            string_domains = sorted(string_domains, key=lambda x: int(x.split('-')[0]))

        #Read in finished, do TSV writing in the correct format
        directory, file_name = os.path.split(chainsaw_annotation_tsv)
        new_file_name = file_name.replace('.tsv', '_prot2DFormattedDomains.tsv')   
        domain_annotation_formatted_file = os.path.join(output_dir, new_file_name)
        
        with open(domain_annotation_formatted_file, 'w', newline='', encoding='utf-8') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            tsv_writer.writerow(['chain_id', 'domain', 'domain_start', 'domain_end'])
            
            for idx, domain in enumerate(string_domains, start=1):
                start, end = map(int, domain.split('-'))
                tsv_writer.writerow([chain_id+"_chain_A", idx, start, end])

        return domain_annotation_formatted_file

def add_header_to_predicted_pdb(pdb_file):
    """
    Method for adding header line to a given pdb_file if none can be found. This is necessary for DSSP (secondary structure assignment) to work properly.
    
    Args:

    - pdb_file (str): Path to pdb_file.

    Returns:

    - the path to the pdb_file Header.
    """
    
    header_line = "added    for DSSP"
    with open(pdb_file, 'r') as input_file:
        pdb_content = input_file.read()

    # Check if the header already exists
    if not pdb_content.startswith("HEADER"):
        #Header does not exist, add it
        updated_pdb_content = f"HEADER    {header_line}\n{pdb_content}"
        with open(pdb_file, 'w') as output_file:
            output_file.write(updated_pdb_content)
        
        return output_file, True  #header was added
    else:
        #Header already exists
        return pdb_file, False  #header was not added

def create_own_annotation_files_for_dir(chainsaw_output_dir, output_directory):
    
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(chainsaw_output_dir):
        if filename.endswith(".tsv"):
            input_filepath = os.path.join(chainsaw_output_dir, filename)
            format_domain_annotation_file_chainsaw_discSplit(input_filepath, output_directory)

def add_header_to_pdb_dir(input_directory):
    #Iterate through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".pdb"):
            input_filepath = os.path.join(input_directory, filename)
            print(input_filepath)
            output_file, header_added = add_header_to_predicted_pdb(input_filepath)

            if header_added:
                print(f"Header added successfully to {input_filepath}")
            else:
                print(f"Header already exists in {input_filepath}")

def extract_sequence_from_pdb(pdb_file):
    """
    Method for extracting the aminoacid sequence from a given pdb-file

    Args:

    - pdb_file (str): Path to pdb_file.

    Returns:

    - the aminoacid sequence of the given protein
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    #Extract sequence from the first chain (you may modify as needed)
    sequence = ""
    for model in structure:
        for chain in model:
            sequence += PDB.PPBuilder().build_peptides(chain)[0].get_sequence()
    
    return sequence

def get_pdb_files_for_id_list(id_list, output_dir):
    """
    method for downloading pdb-files from pdb-website.

    Args:

    - id_list (list): list of pdb_ids than shall be downloaded
    - output_dir (str): Path to directory where the downloaded pdb-files will be saved.

    Returns:

    - Nothing but saves wanted pdb files in output dir if found in the PDB.
    """
    
    #Disable SSL certificate verification
    ssl._create_default_https_context = ssl._create_unverified_context
    for pdb_id in tqdm(id_list, desc="Downloading PDB files"):
        output_file_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        if not os.path.exists(output_file_path):
            try:
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                
                wget.download(pdb_url, out=output_file_path)
            except Exception as e:
                print(f"Failed to download {pdb_id}.pdb: {str(e)}")
        else:
            print(f"{pdb_id}.pdb already exists. Skipping download.")

def calculate_positive_translation_vector(pdb_file, buffer=50):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    min_x = float('inf')
    min_y = float('inf')

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, _ = atom.coord
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y

    shift_x = buffer - min_x
    shift_y = buffer - min_y
    shift_x = round(shift_x, 3)
    shift_y = round(shift_y, 3)
    return (shift_x, shift_y, 0.0)  #z remains unchanged
