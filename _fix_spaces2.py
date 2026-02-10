11# -*- coding: utf-8 -*-
"""Limpieza masiva de espacios residuales en archivos .py tras eliminación de emojis."""
import re
import os

PROJECT = os.path.dirname(os.path.abspath(__file__))
SKIP = {'.venv', '.git', '__pycache__', 'venv', 'node_modules'}

def fix_leading_spaces_in_strings(content):
    """
    Corrige patrones como:
      logger.info(" Texto")  ->  logger.info("Texto")
      print(" Texto")        ->  print("Texto")
      print(f" Texto")       ->  print(f"Texto")
      print(f"\\n Texto")    ->  print(f"\\nTexto")
    Solo dentro de strings que van a logger/print.
    """
    # Patron: comilla (simple o doble), opcionalmente precedida de f/F,
    # seguida de espacio y luego texto que empieza con mayuscula, *, { o letra
    # Tambien cubre \n seguido de espacio
    
    # 1. Quitar espacio al inicio de string: ("  Texto") -> ("Texto")
    content = re.sub(r'((?:logger\.\w+|print)\s*\(\s*f?["\'])\s([A-Z*{\\])', r'\1\2', content)
    
    # 2. Quitar espacio despues de \n dentro de strings de print/logger
    content = re.sub(r'((?:logger\.\w+|print)\s*\(\s*f?["\'].*?\\n)\s([A-Z*{])', r'\1\2', content)
    
    return content

def process_file(filepath):
    """Procesa un archivo .py y retorna numero de cambios."""
    with open(filepath, 'r', encoding='utf-8') as f:
        original = f.read()
    
    modified = fix_leading_spaces_in_strings(original)
    
    if modified != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified)
        # Contar diferencias
        orig_lines = original.split('\n')
        mod_lines = modified.split('\n')
        changes = sum(1 for a, b in zip(orig_lines, mod_lines) if a != b)
        return changes
    return 0

def main():
    total_files = 0
    total_changes = 0
    
    for root, dirs, files in os.walk(PROJECT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if f.endswith('.py') and f != '_fix_spaces2.py':
                path = os.path.join(root, f)
                changes = process_file(path)
                if changes > 0:
                    rel = os.path.relpath(path, PROJECT)
                    print(f"  {rel}: {changes} lineas corregidas")
                    total_files += 1
                    total_changes += changes
    
    print(f"\nTotal: {total_files} archivos, {total_changes} lineas corregidas")

if __name__ == '__main__':
    print("=== Limpieza de espacios residuales en .py ===\n")
    main()
    print("\n=== Listo ===")
