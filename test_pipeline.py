import unittest
import pandas as pd
import numpy as np
from feature_builder import FeatureBuilder

import unittest
import pandas as pd
import numpy as np
from feature_builder import FeatureBuilder

class TestSBAFeaturePipeline(unittest.TestCase):
    def setUp(self):
        """每個測試開始前初始化，確保 FeatureBuilder 實作了 fit/transform"""
        self.builder = FeatureBuilder()

    def test_imputation_and_cleaning(self):
        """測試：缺失值填充與金額清理邏輯"""
        # 建立模擬資料列 (row)
        input_df = pd.DataFrame({
            'SBA_Appv': ['$100,000'],    # 測試金額清理與 log1p
            'RevLineCr': [np.nan],       # 測試 0 缺失值補 'nan'
            'LowDoc': [np.nan],          # 測試 0 缺失值補 'nan'
            'NewExist': [np.nan],        # 測試 NewExist 填充 (若你已在 Builder 加入)
            'State': [np.nan]            # 測試 State 填充 (若你已在 Builder 加入)
        })

        output_df = self.builder.transform(input_df)

        # 1. 驗證金額轉為 log1p(100000)
        expected_money = np.log1p(100000)
        self.assertAlmostEqual(output_df['SBA_Appv'].iloc[0], expected_money, places=4)

        # 2. 驗證缺失值填充字串
        self.assertEqual(output_df['RevLineCr'].iloc[0], "nan")
        self.assertEqual(output_df['LowDoc'].iloc[0], "nan")

    def test_column_lifecycle(self):
        """測試：欄位 (column) 的轉換與刪除生命週期"""
        input_df = pd.DataFrame({
            'NAICS': ['236115'],             # 應變為 NAICS_Section 並刪除原欄位
            'ApprovalDate': ['01-Jan-06'],    # 應變為 Days_Since_Appv 並刪除原欄位
            'FranchiseCode': ['1'],           # 應變為 FranchiseCode_Binary 並刪除原欄位
            'LoanNr_ChkDgt': ['123'],        # 若你在 Builder 有刪除邏輯，應消失
            'Name': ['Test Corp']            # 若你在 Builder 有刪除邏輯，應消失
        })

        output_df = self.builder.transform(input_df)

        # 1. 驗證新欄位產生
        self.assertIn('NAICS_Section', output_df.columns)
        self.assertIn('Days_Since_Appv', output_df.columns)
        self.assertIn('FranchiseCode_Binary', output_df.columns)
        
        # 2. 驗證舊欄位刪除
        self.assertNotIn('NAICS', output_df.columns)
        self.assertNotIn('ApprovalDate', output_df.columns)
        self.assertNotIn('FranchiseCode', output_df.columns)

    def test_date_handling_robustness(self):
        """測試：日期格式異常時的穩定性"""
        # 測試空日期是否正確補為 -1
        input_df = pd.DataFrame({'ApprovalDate': [np.nan]})
        output_df = self.builder.transform(input_df)
        
        self.assertEqual(output_df['Days_Since_Appv'].iloc[0], -1)

if __name__ == '__main__':
    unittest.main()