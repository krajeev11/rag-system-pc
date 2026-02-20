"""
System Document Scanner Module
Scans the Ubuntu/Linux system for documents and indexes them
"""

import os
from pathlib import Path
from typing import List, Set, Optional
import mimetypes
from datetime import datetime


class DocumentScanner:
    """Scans the system for documents and prepares them for indexing"""
    
    def __init__(
        self,
        scan_directories: Optional[List[str]] = None,
        exclude_directories: Optional[List[str]] = None,
        max_file_size_mb: int = 50,
        supported_extensions: Optional[Set[str]] = None
    ):
        """
        Initialize the document scanner
        
        Args:
            scan_directories: List of directories to scan (default: common user directories)
            exclude_directories: Directories to exclude from scanning
            max_file_size_mb: Maximum file size in MB to process
            supported_extensions: Set of file extensions to include
        """
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        # Default supported extensions
        if supported_extensions is None:
            self.supported_extensions = {
                '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml',
                '.csv', '.log', '.conf', '.cfg', '.ini', '.yaml', '.yml',
                '.rst', '.tex', '.org', '.wiki', '.rtf'
            }
        else:
            self.supported_extensions = supported_extensions
        
        # Default scan directories (user's home directory and common document locations)
        if scan_directories is None:
            home = os.path.expanduser("~")
            self.scan_directories = [
                os.path.join(home, "Documents"),
                os.path.join(home, "Downloads"),
                os.path.join(home, "Desktop"),
                os.path.join(home, "Pictures"),
                os.path.join(home, "Videos"),
                os.path.join(home, "Music"),
            ]
            # Filter out non-existent directories
            self.scan_directories = [d for d in self.scan_directories if os.path.exists(d)]
        else:
            self.scan_directories = scan_directories
        
        # Default exclude directories
        if exclude_directories is None:
            self.exclude_directories = {
                '/proc', '/sys', '/dev', '/run', '/tmp', '/var/log',
                '/boot', '/lost+found', '.git', '__pycache__', 'node_modules',
                '.venv', 'venv', 'env', '.env', '.cache', '.local/share/Trash'
            }
        else:
            self.exclude_directories = set(exclude_directories)
    
    def should_exclude(self, file_path: str) -> bool:
        """Check if a file or directory should be excluded"""
        path_str = str(file_path)
        
        # Check if path contains any excluded directory
        for exclude in self.exclude_directories:
            if exclude in path_str:
                return True
        
        # Check if it's a hidden file/directory (except if explicitly included)
        parts = Path(file_path).parts
        if any(part.startswith('.') and part not in ['.', '..'] for part in parts):
            # Allow some common hidden files
            if not any(part in ['.bashrc', '.profile', '.gitignore', '.env'] for part in parts):
                return True
        
        return False
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Check if file is supported and within size limits"""
        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                return False
        except (OSError, PermissionError):
            return False
        
        return True
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[dict]:
        """
        Scan a directory for documents
        
        Args:
            directory: Directory path to scan
            recursive: Whether to scan recursively
            
        Returns:
            List of file information dictionaries
        """
        files_found = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            print(f"Warning: Directory does not exist: {directory}")
            return files_found
        
        if self.should_exclude(str(directory_path)):
            return files_found
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in directory_path.glob(pattern):
                if file_path.is_file():
                    if self.should_exclude(str(file_path)):
                        continue
                    
                    if self.is_supported_file(file_path):
                        try:
                            file_info = {
                                'path': str(file_path),
                                'name': file_path.name,
                                'size': file_path.stat().st_size,
                                'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                                'extension': file_path.suffix.lower()
                            }
                            files_found.append(file_info)
                        except (OSError, PermissionError) as e:
                            continue
        
        except PermissionError:
            print(f"Permission denied: {directory}")
        except Exception as e:
            print(f"Error scanning {directory}: {e}")
        
        return files_found
    
    def scan_system(self, recursive: bool = True) -> List[dict]:
        """
        Scan all configured directories for documents
        
        Args:
            recursive: Whether to scan recursively
            
        Returns:
            List of all found files with metadata
        """
        all_files = []
        
        print(f"Scanning {len(self.scan_directories)} directory(ies)...")
        
        for directory in self.scan_directories:
            print(f"  Scanning: {directory}")
            files = self.scan_directory(directory, recursive=recursive)
            all_files.extend(files)
            print(f"    Found {len(files)} files")
        
        print(f"\nTotal files found: {len(all_files)}")
        return all_files
    
    def get_file_summary(self, files: List[dict]) -> dict:
        """Get summary statistics about scanned files"""
        if not files:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'by_extension': {},
                'by_directory': {}
            }
        
        total_size = sum(f['size'] for f in files)
        by_extension = {}
        by_directory = {}
        
        for file_info in files:
            ext = file_info['extension'] or 'no_extension'
            by_extension[ext] = by_extension.get(ext, 0) + 1
            
            directory = str(Path(file_info['path']).parent)
            by_directory[directory] = by_directory.get(directory, 0) + 1
        
        return {
            'total_files': len(files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'by_extension': dict(sorted(by_extension.items(), key=lambda x: x[1], reverse=True)),
            'by_directory': dict(sorted(by_directory.items(), key=lambda x: x[1], reverse=True))
        }
