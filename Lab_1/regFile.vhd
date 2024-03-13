LIBRARY IEEE;
USE IEEE.std_logic_1164.all;
USE IEEE.numeric_std.all;

-- this is a 32x32-bit register file for a 32-bit MIPS microprocessor 

ENTITY regFile IS
	-- numerical constants are defined using generics (to avoid magic nums)
	GENERIC(
		addressBits_GEN : INTEGER := 5;  -- 0 -> 31
		dataLength_GEN  : INTEGER := 32; -- 32-bit data
		numOfReg_GEN    : INTEGER := 32  -- number of registers = 2 ^ addressBits	
	);	
	PORT(
		-- input signals
		RsSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0); -- read reg 1
		RtSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0); -- read reg 2
		RdSel_IN  : IN  STD_LOGIC_VECTOR((addressBits_GEN - 1) DOWNTO 0); -- write reg
		DataW_IN  : IN  STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0); -- write this data in Rd

		-- clock + control signals
		WrtEN     : IN  STD_LOGIC; -- write enable
		CLK       : IN  STD_LOGIC; -- clock (write on falling and read on rising)		

		-- output signals
		DataRs_OUT : OUT STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0); -- data written in Rs
		DataRt_OUT : OUT STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0)  -- data written in Rd
	);

END regFile;


ARCHITECTURE regFile_ARCH OF regFile IS
	-- define a type as a 1D array of numOfReg elements 
	TYPE regFile_TYP IS ARRAY(0 TO (numOfReg_GEN - 1)) OF STD_LOGIC_VECTOR((dataLength_GEN - 1) DOWNTO 0);
	-- define registerFile signal to hold register data as a 1D array of numOfReg register elements
BEGIN
	PROCESS(CLK, WrtEN) IS
		-- define registerFile variable to hold register data as a 1D array of numOfReg register elements       
		VARIABLE registerFile_VAR : regFile_TYP := (
							X"00000000",
							X"00000001",
							X"00000002",
							X"00000003",
							X"00000004",
							X"00000005",
							X"00000006",
							X"00000007",
							X"00000008",
							X"00000009",
							X"0000000A",
							X"0000000B",
							X"0000000C",
							X"0000000D",
							X"0000000E",
							X"0000000F",
							X"00000010",
							X"00000011",
							X"00000012",
							X"00000013",
							X"00000014",
							X"00000015",
							X"00000016",
							X"00000017",
							X"00000018",
							X"00000019",
							X"0000001A",
							X"0000001B",
							X"0000001C",
							X"0000001D",
							X"0000001E",
							X"0000001F"
							);
	BEGIN	
		IF (FALLING_EDGE(CLK) AND WrtEN = '1') THEN -- write data in Rd on falling edge && @ WrtEN = 1
			IF (RdSel_IN /= "00000") THEN -- if destination is NOT reg_zero, write 
				registerFile_VAR(TO_INTEGER(UNSIGNED(RdSel_IN))) := DataW_IN;
			END IF;
		END IF;
		-- read Rs && Rt
		IF (RISING_EDGE(CLK)) THEN
			DataRs_OUT <= registerFile_VAR(TO_INTEGER(UNSIGNED(RsSel_IN)));
			DataRt_OUT <= registerFile_VAR(TO_INTEGER(UNSIGNED(RtSel_IN)));
		END IF;
	END PROCESS;
END regFile_ARCH;