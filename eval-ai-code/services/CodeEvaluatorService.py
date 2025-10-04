from dataclasses import dataclass
from openai import AsyncAzureOpenAI
import aiofiles
import asyncio
import json
import click
from pathlib import Path
from typing import List, Set


@dataclass
class FileEvaluation:
    """Data class to store evaluation results for a single file."""
    filename: str
    score: int
    reason: str
    file_type: str


@dataclass
class OverallEvaluation:
    """Data class to store overall evaluation results."""
    score: int
    reason: str
    total_files: int
    file_evaluations: List[FileEvaluation]

def print_results(evaluation: OverallEvaluation):
    """Print the evaluation results in a formatted way."""
    click.echo("\n" + "="*80)
    click.echo("AI CODE GENERATION EVALUATION RESULTS")
    click.echo("="*80)
    
    # Overall score
    click.echo(f"\nOVERALL SCORE: {evaluation.score}/10")
    click.echo(f"REASON: {evaluation.reason}")
    click.echo(f"TOTAL FILES ANALYZED: {evaluation.total_files}")
    
    if evaluation.file_evaluations:
        click.echo("\nINDIVIDUAL FILE SCORES:")
        click.echo("-" * 80)
        
        # Sort by score (highest first)
        sorted_evals = sorted(evaluation.file_evaluations, key=lambda x: x.score, reverse=True)
        
        for eval in sorted_evals:
            click.echo(f"\nFile: {eval.filename}")
            click.echo(f"Score: {eval.score}/10")
            click.echo(f"Type: {eval.file_type}")
            click.echo(f"Reason: {eval.reason}")
            click.echo("-" * 40)

class AICodeEvaluator:
    """Main class for evaluating code files using Azure OpenAI."""

    # Common code file extensions
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.cs',
        '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.sh',
        '.ps1', '.sql', '.html', '.css', '.vue', '.dart', '.r', '.m'
    }

    def __init__(self, azure_endpoint: str, api_key: str, api_version: str = "2024-02-15-preview"):
        """Initialize the evaluator with Azure OpenAI credentials."""
        self.client: AsyncAzureOpenAI = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )

    def is_code_file(self, file_path: Path, exclude_extensions: Set[str] = None) -> bool:
        """Check if a file is a code file based on its extension."""
        if exclude_extensions and file_path.suffix.lower() in exclude_extensions:
            return False
        return file_path.suffix.lower() in self.CODE_EXTENSIONS

    def should_exclude_folder(self, folder_path: Path, exclude_folders: Set[str] = None) -> bool:
        """Check if a folder should be excluded."""
        if exclude_folders:
            # Check if any part of the path matches excluded folders
            for part in folder_path.parts:
                if part in exclude_folders:
                    return True
        return False

    def get_code_files(self, folder_path: Path, exclude_extensions: Set[str] = None, exclude_folders: Set[str] = None) -> List[Path]:
        """Recursively find all code files in the given folder."""
        code_files = []

        # Default exclusions
        default_exclusions = {'.git', 'node_modules', '__pycache__', 'venv', 'env', '.venv'}

        # Combine default exclusions with user-specified exclusions
        all_excluded_folders = default_exclusions
        if exclude_folders:
            all_excluded_folders = all_excluded_folders.union(exclude_folders)

        # Use rglob for recursive globbing which is more efficient
        for file_path in folder_path.rglob('*'):
            # Skip files in excluded directories
            if any(part.startswith('.') or part in all_excluded_folders
                   for part in file_path.parts):
                continue

            # Skip if the parent folder should be excluded
            if self.should_exclude_folder(file_path.parent, exclude_folders):
                continue

            if (file_path.is_file() and
                self.is_code_file(file_path, exclude_extensions) and
                file_path.stat().st_size > 0):
                code_files.append(file_path)

        return code_files

    async def read_file_content(self, file_path: Path) -> str:
        """Asynchronously read and return the content of a file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
                # Limit content size to avoid token limits
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
                return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    async def evaluate_file(self, file_path: Path, deployment_name: str, base_path: Path) -> FileEvaluation:
        """Asynchronously evaluate a single file using Azure OpenAI."""
        content = await self.read_file_content(file_path)

        prompt = f"""
        Analyze the following code and determine how likely it is that this code was generated by an AI code generation tool (like GitHub Copilot, ChatGPT, Claude, etc.).

        Consider these factors:
        1. Code style and patterns (AI often generates very consistent, sometimes overly structured code)
        2. Comments (AI tends to generate comprehensive comments, sometimes overly detailed)
        3. Variable and function naming (AI often uses very descriptive, sometimes verbose names)
        4. Code structure (AI tends to follow best practices rigidly)
        5. Error handling (AI often includes comprehensive error handling)
        6. Documentation strings and type hints (AI frequently includes these)
        7. Coding patterns that are typical of AI generation
        8. Lack of personal coding quirks or shortcuts that human developers often use

        File: {file_path.name}
        File type: {file_path.suffix}

        Code:
        ```
        {content}
        ```

        Provide a score from 1 to 10 where:
        - 1-2: Very unlikely to be AI-generated (clearly human-written)
        - 3-4: Probably human-written with some AI assistance possible
        - 5-6: Could be either human or AI-written
        - 7-8: Likely AI-generated with possible human modifications
        - 9-10: Very likely AI-generated

        Respond with only a JSON object in this format:
        {{"score": <number>, "reason": "<detailed explanation of your reasoning>"}}
        """

        try:
            response = await self.client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert code analyst specializing in identifying AI-generated code. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            result_text = response.choices[0].message.content.strip()

            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                score = max(1, min(10, int(result.get('score', 5))))
                reason = result.get('reason', 'No reason provided')
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                score = 5
                reason = f"Unable to parse AI response: {result_text[:200]}"

            return FileEvaluation(
                filename=str(file_path.relative_to(base_path)),
                score=score,
                reason=reason,
                file_type=file_path.suffix
            )

        except Exception as e:
            return FileEvaluation(
                filename=str(file_path.relative_to(base_path)),
                score=5,
                reason=f"Error during evaluation: {str(e)}",
                file_type=file_path.suffix
            )

    def calculate_overall_score(self, file_evaluations: List[FileEvaluation]) -> OverallEvaluation:
        """Calculate overall evaluation based on individual file scores."""
        if not file_evaluations:
            return OverallEvaluation(
                score=1,
                reason="No code files found to evaluate",
                total_files=0,
                file_evaluations=[]
            )

        # Calculate weighted average (give more weight to files with higher scores)
        total_score = sum(eval.score for eval in file_evaluations)
        average_score = total_score / len(file_evaluations)

        # Round to nearest integer
        overall_score = max(1, min(10, round(average_score)))

        # Generate reason based on score distribution
        high_scores = sum(1 for eval in file_evaluations if eval.score >= 7)
        medium_scores = sum(1 for eval in file_evaluations if 4 <= eval.score <= 6)
        low_scores = sum(1 for eval in file_evaluations if eval.score <= 3)

        if high_scores > len(file_evaluations) * 0.6:
            reason = f"Most files ({high_scores}/{len(file_evaluations)}) show strong indicators of AI generation"
        elif low_scores > len(file_evaluations) * 0.6:
            reason = f"Most files ({low_scores}/{len(file_evaluations)}) appear to be human-written"
        else:
            reason = f"Mixed results: {high_scores} likely AI-generated, {medium_scores} uncertain, {low_scores} likely human-written"

        return OverallEvaluation(
            score=overall_score,
            reason=reason,
            total_files=len(file_evaluations),
            file_evaluations=file_evaluations
        )

    async def evaluate_folder(self, folder_path: Path, deployment_name: str, exclude_extensions: Set[str] = None, exclude_folders: Set[str] = None) -> OverallEvaluation:
        """Asynchronously evaluate all code files in a folder and its subfolders."""
        click.echo(f"Scanning for code files in: {folder_path}")

        if exclude_extensions:
            click.echo(f"Excluding file types: {', '.join(sorted(exclude_extensions))}")
        if exclude_folders:
            click.echo(f"Excluding folders: {', '.join(sorted(exclude_folders))}")

        code_files = self.get_code_files(folder_path, exclude_extensions, exclude_folders)

        if not code_files:
            click.echo("No code files found!")
            return OverallEvaluation(
                score=1,
                reason="No code files found to evaluate",
                total_files=0,
                file_evaluations=[]
            )

        click.echo(f"Found {len(code_files)} code files to evaluate (including subfolders)...")

        # Create tasks for concurrent evaluation
        tasks = []
        for file_path in code_files:
            task = self.evaluate_file(file_path, deployment_name, folder_path)
            tasks.append(task)

        # Use a semaphore to limit concurrent requests to avoid rate limits
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests

        async def evaluate_with_semaphore(task):
            async with semaphore:
                return await task

        # Execute all tasks concurrently with progress tracking
        file_evaluations = []
        with click.progressbar(length=len(tasks), label='Evaluating files') as bar:
            for task in asyncio.as_completed([evaluate_with_semaphore(task) for task in tasks]):
                evaluation = await task
                file_evaluations.append(evaluation)
                bar.update(1)

        return self.calculate_overall_score(file_evaluations)