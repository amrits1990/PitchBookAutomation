#!/usr/bin/env python3
"""
Update ROA, ROIC, and Net_ROIC formulas in database to use NOPAT instead of net_income
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import FinancialDatabase
from virtual_fields import DEFAULT_RATIOS, TAX_RATE

def update_ratio_formulas():
    """Update the three ratio formulas in database"""
    print("🔄 UPDATING RATIO FORMULAS IN DATABASE")
    print("=" * 60)
    
    # Ratios to update
    ratios_to_update = ['ROA', 'ROIC', 'Net_ROIC']
    
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            for ratio_name in ratios_to_update:
                print(f"\n📊 Updating {ratio_name}...")
                
                # Get new formula from virtual_fields.py
                new_definition = DEFAULT_RATIOS.get(ratio_name)
                if not new_definition:
                    print(f"   ❌ Definition not found in virtual_fields.py")
                    continue
                
                new_formula = new_definition['formula']
                new_description = new_definition['description']
                
                print(f"   Old formula: net_income / [denominator]")
                print(f"   New formula: {new_formula}")
                
                # Update in database
                cursor.execute("""
                    UPDATE ratio_definitions 
                    SET 
                        formula = %s,
                        description = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE name = %s AND is_active = true
                """, (new_formula, new_description, ratio_name))
                
                if cursor.rowcount > 0:
                    print(f"   ✅ Updated successfully ({cursor.rowcount} rows)")
                else:
                    print(f"   ⚠️  No rows updated - ratio may not exist")
            
            # Commit all changes
            db.connection.commit()
            print(f"\n💾 All changes committed to database")

def verify_updates():
    """Verify that the updates were successful"""
    print(f"\n🔍 VERIFYING UPDATES")
    print("=" * 40)
    
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            cursor.execute("""
                SELECT name, formula, description
                FROM ratio_definitions 
                WHERE name IN ('ROA', 'ROIC', 'Net_ROIC') AND is_active = true
                ORDER BY name
            """)
            
            results = cursor.fetchall()
            
            for name, formula, description in results:
                print(f"\n📈 {name}:")
                print(f"   Formula: {formula}")
                
                # Check if update was successful
                if 'ebit' in formula.lower() and 'net_income' not in formula.lower():
                    print(f"   ✅ SUCCESS: Now using NOPAT formula")
                elif 'net_income' in formula.lower():
                    print(f"   ❌ FAILED: Still using old net_income formula")
                else:
                    print(f"   ❓ UNKNOWN: Unexpected formula pattern")

def reinitialize_ratios_option():
    """Alternative: Reinitialize all default ratios (nuclear option)"""
    print(f"\n🔄 ALTERNATIVE: REINITIALIZE ALL RATIOS")
    print("=" * 50)
    
    print("If the update approach doesn't work, you can reinitialize all ratios:")
    print("This will update ALL ratio definitions to match virtual_fields.py")
    
    response = input("Do you want to reinitialize all ratios? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            from ratio_manager import initialize_default_ratios
            
            print("🔄 Reinitializing all default ratios...")
            count = initialize_default_ratios(created_by="formula_update")
            print(f"✅ Reinitialized {count} ratio definitions")
            
            # Verify the specific ratios
            verify_updates()
            
        except Exception as e:
            print(f"❌ Reinitialization failed: {e}")
    else:
        print("Skipping reinitialization")

def clear_calculated_ratios():
    """Clear calculated ratios so they get recalculated with new formulas"""
    print(f"\n🗑️  CLEARING CALCULATED RATIOS")
    print("=" * 40)
    
    print("After updating formulas, you should recalculate all ratios")
    print("This will delete existing calculated ratios for ROA, ROIC, Net_ROIC")
    
    response = input("Clear calculated ratios for these 3 ratios? (y/n): ").strip().lower()
    
    if response == 'y':
        with FinancialDatabase() as db:
            with db.connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM calculated_ratios 
                    WHERE ratio_name IN ('ROA', 'ROIC', 'Net_ROIC')
                """)
                
                deleted_count = cursor.rowcount
                db.connection.commit()
                
                print(f"🗑️  Deleted {deleted_count} calculated ratio records")
                print(f"✅ These ratios will be recalculated with new formulas when next calculated")
    else:
        print("Keeping existing calculated ratios")

def main():
    """Main update function"""
    print("🔬 RATIO FORMULA UPDATER")
    print("=" * 80)
    print("Updating ROA, ROIC, Net_ROIC to use NOPAT instead of net_income")
    print("=" * 80)
    
    try:
        # Method 1: Direct update
        update_ratio_formulas()
        verify_updates()
        
        # Method 2: Alternative reinitialize option
        reinitialize_ratios_option()
        
        # Clean up calculated ratios
        clear_calculated_ratios()
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. ✅ Ratio definitions updated in database")
        print(f"2. 🔄 Run ratio calculations for companies to get new values")
        print(f"3. 🧪 Test that new ratios use NOPAT instead of net_income")
        print(f"4. 📊 Verify ratio values are different from before")
        
        print(f"\n💡 TO RECALCULATE RATIOS:")
        print(f"   from simple_ratio_calculator import SimpleRatioCalculator")
        print(f"   with SimpleRatioCalculator() as calc:")
        print(f"       calc.calculate_company_ratios('AAPL')  # Example")
        
    except Exception as e:
        print(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()