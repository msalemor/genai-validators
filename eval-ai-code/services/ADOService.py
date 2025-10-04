import os
import tempfile
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from urllib.parse import urlparse, parse_qs

def download_pr_changes(pr_url, pat_token=None)-> str:
    """
    Download modified code from a PR to a temporary folder maintaining file structure.
    
    Args:
        pr_url (str): Azure DevOps PR URL
        pat_token (str): Personal Access Token
    
    Returns:
        str: Path to temporary folder containing the changes
    """
    # Parse PR URL to extract organization, project, repo, and PR ID
    parsed_url = urlparse(pr_url)
    path_parts = parsed_url.path.strip('/').split('/')
    
    organization = parsed_url.netloc.split('.')[0]
    project = path_parts[0]
    repo_name = path_parts[2]
    pr_id = int(path_parts[4])
    
    # Create connection to Azure DevOps
    credentials = BasicAuthentication('', pat_token)
    connection = Connection(base_url=f'https://dev.azure.com/{organization}', creds=credentials)
    
    # Get clients
    git_client = connection.clients.get_git_client()
    
    # Get PR details
    pr = git_client.get_pull_request(repository_id=repo_name, pull_request_id=pr_id, project=project)
    
    # Get PR changes
    changes = git_client.get_pull_request_iteration_changes(
        repository_id=repo_name, 
        pull_request_id=pr_id, 
        iteration_id=1,
        project=project
    )
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Download modified files
    for change in changes.change_entries:
        if change.change_type in ['add', 'edit']:
            file_path = change.item.path
            
            # Get file content from the source branch
            file_content = git_client.get_item_content(
                repository_id=repo_name,
                path=file_path,
                version_descriptor={'version': pr.source_ref_name, 'version_type': 'branch'},
                project=project
            )
            
            # Create directory structure in temp folder
            full_path = os.path.join(temp_dir, file_path.lstrip('/'))
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file content
            with open(full_path, 'wb') as f:
                f.write(file_content)
    
    return temp_dir