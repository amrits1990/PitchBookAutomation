#!/usr/bin/env python3
"""
Check ratio definitions in database to see if ROA, ROIC, Net_ROIC have old formulas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import FinancialDatabase

def check_ratio_definitions():
    """Check current ratio definitions in database"""
    print("🔍 CHECKING RATIO DEFINITIONS IN DATABASE")
    print("=" * 60)
    
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            # Check ROA, ROIC, Net_ROIC specifically
            cursor.execute("""
                SELECT name, formula, description, category, is_active
                FROM ratio_definitions 
                WHERE name IN ('ROA', 'ROIC', 'Net_ROIC')
                ORDER BY name
            """)
            
            results = cursor.fetchall()
            
            if results:
                print(f"📊 Found {len(results)} ratio definitions:")
                for name, formula, description, category, is_active in results:
                    print(f"\n📈 {name}:")
                    print(f"   Formula: {formula}")
                    print(f"   Description: {description}")
                    print(f"   Category: {category}")
                    print(f"   Active: {is_active}")
                    
                    # Check if using old net_income formula
                    if 'net_income' in formula.lower():
                        print(f"   ⚠️  ISSUE: Still using old net_income formula!")
                    elif 'ebit' in formula.lower():
                        print(f"   ✅ GOOD: Using NOPAT (EBIT) formula")
                    else:
                        print(f"   ❓ UNKNOWN: Formula doesn't match expected pattern")
            else:
                print(f"❌ No ratio definitions found for ROA, ROIC, Net_ROIC")
                print(f"   This might mean ratio definitions haven't been initialized yet")
            
            # Check all profitability ratios
            print(f"\n📋 ALL PROFITABILITY RATIOS:")
            cursor.execute("""
                SELECT name, formula
                FROM ratio_definitions 
                WHERE category = 'profitability' AND is_active = true
                ORDER BY name
            """)
            
            prof_results = cursor.fetchall()
            for name, formula in prof_results:
                print(f"   {name}: {formula}")

def check_virtual_fields_vs_database():
    """Compare virtual_fields.py definitions with database"""
    print(f"\n🔄 COMPARING VIRTUAL_FIELDS.PY vs DATABASE")
    print("=" * 60)
    
    # Import virtual fields
    from virtual_fields import DEFAULT_RATIOS
    
    # Get database definitions
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            cursor.execute("""
                SELECT name, formula
                FROM ratio_definitions 
                WHERE name IN ('ROA', 'ROIC', 'Net_ROIC') AND is_active = true
            """)
            
            db_ratios = {name: formula for name, formula in cursor.fetchall()}
    
    # Compare each ratio
    for ratio_name in ['ROA', 'ROIC', 'Net_ROIC']:
        print(f"\n📊 {ratio_name}:")
        
        # Virtual fields definition
        vf_def = DEFAULT_RATIOS.get(ratio_name, {})
        vf_formula = vf_def.get('formula', 'NOT FOUND')
        print(f"   virtual_fields.py: {vf_formula}")
        
        # Database definition
        db_formula = db_ratios.get(ratio_name, 'NOT FOUND')
        print(f"   database:          {db_formula}")
        
        # Check if they match
        if vf_formula == db_formula:
            print(f"   ✅ MATCH")
        elif vf_formula == 'NOT FOUND':
            print(f"   ❌ NOT FOUND in virtual_fields.py")
        elif db_formula == 'NOT FOUND':
            print(f"   ❌ NOT FOUND in database")
        else:
            print(f"   ⚠️  MISMATCH - Database needs update!")

def main():
    """Main check function"""
    print("🔬 RATIO DEFINITION CHECKER")
    print("=" * 80)
    print("Checking if ROA, ROIC, Net_ROIC have been updated to use NOPAT")
    print("=" * 80)
    
    try:
        check_ratio_definitions()
        check_virtual_fields_vs_database()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. If database has old formulas, run ratio initialization script")
        print(f"2. If ratios are missing, run initialize_default_ratios()")
        print(f"3. If formulas are updated, recalculate ratios for all companies")
        
    except Exception as e:
        print(f"❌ Check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()