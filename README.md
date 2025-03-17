# Speeding_NVLM_Decoder_Inference_Computation
Using Different Frameworks and Methods to Speed Up NVLM Inference
NYCU HPC Team 來自國立陽明交通大學，成功將 NVLM 推理速度提升了 3.3 倍！

* Team Members: Chu-Siang Tseng, Shun-Yu Yang, Cheng-Wei Lin, Chen-Kai Chang, Zong-Hua Wu, Jia-Hui Shen
* NVIDIA Mentor: Shijie Wang

在 NVLM（NVIDIA Vision Language Model） 推理領域的加速研究，具有相當高的發展潛力。NVLM 作為一款先進的多模態模型，結合了文本與圖像的處理能力，在視覺-語言任務（如圖像辨識、文檔分析、視覺問答等）方面展現了領先的成效。若能進一步優化其推理效率，便能顯著降低計算資源與時間成本，讓這些強大的功能得以更快速、更即時地運用在真實場景中。

加速 NVLM 推理之所以受到重視，是因為多模態模型的應用範圍非常廣泛，尤其在需要即時反應的領域（例如自動駕駛、醫療診斷、智慧助理系統等），推理速度的快慢常直接影響系統的可用性與可靠性。讓 NVLM 推理更加高效，不僅能提升多模態應用的實用性，也能促進這類技術在日常生活與產業界的普及化。

目前的研究成果顯示，透過優化模型結構（如採用混合架構、動態高解析度影像處理等），能在確保或甚至提升模型準確度的前提下，大幅加快推理速度，並降低硬體需求與計算成本。這些技術上的突破不僅縮減了資源消耗，也讓多模態應用更易被大眾和業界接受，進一步推動整個 AI 生態系往更高效、更智慧的方向演進。

因應近期 LLM（Large Language Model） 及 LMM（Large Multi-Modal Model） 的快速發展，我們選定了加速 NVLM 推理作為本次 Hackathon 的主題研究。在此過程中，我們針對程式碼進行了多項調整，其中包括修改 generate 函式，並新增「chat batch」功能以處理多筆輸入，達到能平行推理多個問題的目的。藉由這些優化措施，我們成功顯著提升 NVLM 的推理速度，同時兼顧模型的準確度與穩定性。

我們很幸運的也有被[科技報橘](https://buzzorange.com/techorange/2025/03/13/nchc-open-hackathon-nvidia/)報導出來

更多資訊請看：[https://github.com/chu-siang/Speeding_NVLM_Decoder_Inference_Computation](https://github.com/nqobu/nvidia/tree/main/20241204 )
